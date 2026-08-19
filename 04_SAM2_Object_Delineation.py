import os
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import box, shape
from shapely.ops import unary_union
import numpy as np
import torch
import cv2
import warnings

# Core defense 1: Prevent PyTorch VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    raise ImportError("SAM2 library not found. Please install it first.")

warnings.filterwarnings('ignore')

# ================= 1. Global Configurations and Paths =================
print("Starting SAM2 Object Delineation Pipeline...")

#The input here must be the 3m high-resolution PlanetScope imagery, 
tif_path = r'./data/PlanetScope_2025.tif'
geojson_path = r'./results/sam_box_prompts_2025.geojson'
out_shp = r'./results/Bamboo_Object_Distribution_2025.shp'

TILE_SIZE = 1024
OVERLAP = 256
STRIDE = TILE_SIZE - OVERLAP
OFFSET_PASS_2 = STRIDE // 2

CLOSE_DISK_RADIUS = 3
EROSION_DISK_RADIUS = 3

# Core defense 2: Limit prompt batch size to prevent OOM errors
PROMPT_BATCH_SIZE = 16

# ================= 2. Model Initialization =================
checkpoint = "./sam2_hiera_large.pt"
if not os.path.exists(checkpoint):
    print("Downloading SAM2 checkpoint...")
    os.system("wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O ./sam2_hiera_large.pt")

print("Loading SAM2 model...")
sam2_model = build_sam2("sam2_hiera_l.yaml", checkpoint, device="cuda")
sam2_model.to(dtype=torch.bfloat16)
predictor = SAM2ImagePredictor(sam2_model)

# ================= 3. Core Algorithm =================
def run_sliding(src, gdf_prompts, transform, height, width, offset=0):
    global_masks = {}
    rows = range(offset, height, STRIDE)
    cols = range(offset, width, STRIDE)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_DISK_RADIUS * 2 + 1, CLOSE_DISK_RADIUS * 2 + 1))
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EROSION_DISK_RADIUS * 2 + 1, EROSION_DISK_RADIUS * 2 + 1))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for r in rows:
            for c in cols:
                w, h = min(TILE_SIZE, width - c), min(TILE_SIZE, height - r)
                if w < 64 or h < 64:
                    continue

                window = Window(c, r, w, h)
                tile_geom = box(*rasterio.windows.bounds(window, transform))

                sub_prompts = gdf_prompts[gdf_prompts.intersects(tile_geom)]
                if sub_prompts.empty:
                    continue

                tile = src.read(window=window)
                tile = np.transpose(tile, (1, 2, 0))

                # Assuming 4-band image: Blue(0), Green(1), Red(2), NIR(3)
                nir = tile[:, :, 3].astype(np.float32)
                red = tile[:, :, 2].astype(np.float32)
                tile_ndvi = (nir - red) / (nir + red + 1e-8)

                win_transform = src.window_transform(window)

                valid_pts = []
                valid_boxes = []
                valid_species = []

                for _, row in sub_prompts.iterrows():
                    geom = row.geometry
                    species = str(row.get('Species', 'Unknown'))
                    minx, miny, maxx, maxy = geom.bounds

                    px1, py1 = ~win_transform * (minx, miny)
                    px2, py2 = ~win_transform * (maxx, maxy)
                    px_min, px_max = np.clip(sorted([px1, px2]), 0, w-1)
                    py_min, py_max = np.clip(sorted([py1, py2]), 0, h-1)

                    if (px_max - px_min < 1) or (py_max - py_min < 1):
                        continue

                    cx, cy = ~win_transform * (geom.centroid.x, geom.centroid.y)

                    center_ndvi = tile_ndvi[int(np.clip(cy, 0, h-1)), int(np.clip(cx, 0, w-1))]
                    if np.isnan(center_ndvi) or center_ndvi < 0.1:
                        continue

                    EXPAND_PX = 20
                    box_minx = np.clip(px_min - EXPAND_PX, 0, w - 1)
                    box_miny = np.clip(py_min - EXPAND_PX, 0, h - 1)
                    box_maxx = np.clip(px_max + EXPAND_PX, 0, w - 1)
                    box_maxy = np.clip(py_max + EXPAND_PX, 0, h - 1)

                    valid_pts.append([cx, cy])
                    valid_boxes.append([box_minx, box_miny, box_maxx, box_maxy])
                    valid_species.append(species)

                if not valid_pts:
                    continue

                print(f"Tile ({r}, {c}): Found {len(valid_pts)} valid seed points.")

                green = tile[:, :, 1].astype(np.float32)
                tile_false_color = np.zeros((h, w, 3), dtype=np.uint8)

                def stretch_8bit(band):
                    valid_mask = ~np.isnan(band) & ~np.isinf(band)
                    if not np.any(valid_mask):
                        return np.zeros_like(band, dtype=np.uint8)
                    p2, p98 = np.percentile(band[valid_mask], (2, 98))
                    if p98 == p2 or np.isnan(p98):
                        return np.zeros_like(band, dtype=np.uint8)
                    return np.clip((band - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)

                # Create false color composite for SAM2 visual processing
                tile_false_color[:,:,0] = stretch_8bit(nir)
                tile_false_color[:,:,1] = stretch_8bit(red)
                tile_false_color[:,:,2] = stretch_8bit(green)

                predictor.set_image(tile_false_color)

                core_x_min = OVERLAP//2 if c > 0 else 0
                core_x_max = TILE_SIZE - OVERLAP//2 if c + TILE_SIZE < width else w
                core_y_min = OVERLAP//2 if r > 0 else 0
                core_y_max = TILE_SIZE - OVERLAP//2 if r + TILE_SIZE < height else h

                # Core defense 3: Micro-batch processing to prevent memory overflow
                for batch_start in range(0, len(valid_pts), PROMPT_BATCH_SIZE):
                    batch_end = batch_start + PROMPT_BATCH_SIZE

                    pts_array = np.array(valid_pts[batch_start:batch_end], dtype=np.float32)[:, None, :]
                    labels_array = np.ones((len(pts_array), 1), dtype=np.int32)
                    boxes_array = np.array(valid_boxes[batch_start:batch_end], dtype=np.float32)
                    cur_species = valid_species[batch_start:batch_end]

                    masks, scores, _ = predictor.predict(
                        point_coords=pts_array,
                        point_labels=labels_array,
                        box=boxes_array,
                        multimask_output=True
                    )

                    # Handle dimensional squeeze when batch_size is 1
                    if scores.ndim == 1:
                        scores = scores[np.newaxis, ...]
                        masks = masks[np.newaxis, ...]

                    for i, species in enumerate(cur_species):
                        best_idx = np.argmax(scores[i])
                        if scores[i, best_idx] < 0.40:
                            continue

                        raw_mask = masks[i, best_idx].astype(np.uint8)
                        closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)
                        final_optimized_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_ERODE, erosion_kernel)

                        if species not in global_masks:
                            global_masks[species] = np.zeros((height, width), dtype=bool)

                        global_masks[species][r+int(core_y_min):r+int(core_y_max), c+int(core_x_min):c+int(core_x_max)] |= final_optimized_mask[int(core_y_min):int(core_y_max), int(core_x_min):int(core_x_max)].astype(bool)

                # Clear cache after processing each tile
                torch.cuda.empty_cache()

    return global_masks

# ================= 4. Dual-Phase Execution =================
print("\nLoading vector prompt boxes...")
gdf_prompts = gpd.read_file(geojson_path)

with rasterio.open(tif_path) as src:
    transform, crs, width, height = src.transform, src.crs, src.width, src.height
    if gdf_prompts.crs != crs:
        gdf_prompts = gdf_prompts.to_crs(crs)

    print("\nPhase 1: Standard grid extraction...")
    masks_1 = run_sliding(src, gdf_prompts, transform, height, width, offset=0)
    
    print("\nPhase 2: Offset grid extraction...")
    masks_2 = run_sliding(src, gdf_prompts, transform, height, width, offset=OFFSET_PASS_2)

# ================= 5. Spatial Fusion and Conflict Resolution =================
print("\nExecuting spatial fusion and species exclusivity processing...")

# Merge masks from both phases
mask_fr = masks_1.get('F_robusta', np.zeros((height, width), dtype=bool)) | masks_2.get('F_robusta', np.zeros((height, width), dtype=bool))
mask_bf = masks_1.get('B_faberi', np.zeros((height, width), dtype=bool)) | masks_2.get('B_faberi', np.zeros((height, width), dtype=bool))

# Resolve spatial overlap between species
overlap = mask_fr & mask_bf
mask_fr[overlap] = False
mask_bf[overlap] = False
final_masks = {'F_robusta': mask_fr, 'B_faberi': mask_bf}

polygons = []
for sp, final_mask in final_masks.items():
    temp_geoms = [shape(geom_dict) for geom_dict, val in shapes(final_mask.astype(np.uint8), transform=transform) if val == 1]
    if not temp_geoms:
        continue

    united_geometry = unary_union(temp_geoms)
    if united_geometry.geom_type == 'Polygon':
        polygons.append({'properties': {'Species': sp}, 'geometry': united_geometry})
    elif united_geometry.geom_type == 'MultiPolygon':
        polygons.extend([{'properties': {'Species': sp}, 'geometry': g} for g in united_geometry.geoms])

if polygons:
    gdf_out = gpd.GeoDataFrame.from_features(polygons, crs=crs)
    gdf_out = gdf_out[gdf_out.geometry.area > 5]
    os.makedirs(os.path.dirname(out_shp), exist_ok=True)
    gdf_out.to_file(out_shp)
    print(f"\nExtraction complete. Results saved to: {out_shp}")
else:
    print("\nExtraction finished. No valid boundaries were obtained.")
