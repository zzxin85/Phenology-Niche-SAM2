// ==========================================
// File: 02_GEE_Export_Composite_Image.js
// Purpose: Generate 9-band composite TIFF for Python TW-DTW processing
// Band 0-6: NDVI (Month 4 to 10), Band 7: DEM, Band 8: Mask
// ==========================================

// 1. Define Study Area and Time
// Replace with your Wolong National Nature Reserve boundary asset
var roi = ee.FeatureCollection("users/account/wolong_boundary"); 
var year = 2025;
var startDate = year + '-01-01';
var endDate = year + '-12-31';

// 2. Sentinel-2 Cloud Masking Function
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  // Mask out cloudy and cirrus pixels
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

// 3. Process Sentinel-2 to NDVI
var s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .map(maskS2clouds)
  .map(function(img) {
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    return img.addBands(ndvi);
  });

// 4. Generate 7-band NDVI Time Series (April to October)
// These 7 months correspond to DOY 105 to 285 in your Python script
var months = ee.List.sequence(4, 10);
var ndviBands = ee.ImageCollection.fromImages(months.map(function(m) {
  return s2Col.filter(ee.Filter.calendarRange(m, m, 'month'))
              .select('NDVI')
              .median() // Monthly median composite
              .rename(ee.String('NDVI_').cat(ee.Number(m).format('%d')));
})).toBands();

// Rename bands strictly to NDVI_1 ... NDVI_7 for clean export
var cleanNames = ee.List.sequence(1, 7).map(function(n) { 
  return ee.String('NDVI_').cat(ee.Number(n).format('%d')); 
});
var ndviStack = ndviBands.rename(cleanNames);

// 5. NASADEM Elevation Data
var dem = ee.Image('NASA/NASADEM_HGT/001')
  .select('elevation')
  .rename('DEM');

// 6. Generate Background Mask
// 1 = Valid target (vegetation), 0 = Background
// You can also add Sentinel-1 VH mask here if needed
var mask = ndviStack.select('NDVI_4').gt(0.2).rename('Mask');

// 7. Combine All Bands into a 9-Band Composite
// Bands 0-6: NDVI_1 to NDVI_7
// Band 7: DEM
// Band 8: Mask
var finalComposite = ndviStack.addBands(dem).addBands(mask).clip(roi);

// 8. Export to Google Drive
Export.image.toDrive({
  image: finalComposite,
  description: 'Composite_Input_2025', // Task name
  folder: 'Bamboo_Mapping_Data', // Folder in your Google Drive
  scale: 10, // 10m spatial resolution
  region: roi.geometry(),
  maxPixels: 1e13, // Allow large export
  fileFormat: 'GeoTIFF'
});
