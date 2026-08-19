// ==========================================
// 1. Inputs and Parameters
// ==========================================
// Replace this with your uploaded sample points asset path
// Ensure the shapefile has a 'Species' column with values 'fr' or 'bf'
var samplePoints = ee.FeatureCollection("users/account/bamboo_samples");

// Define the time window for the study year
var startDate = '2025-01-01';
var endDate = '2025-12-31';

// ==========================================
// 2. Cloud Masking and NDVI Calculation
// ==========================================
// Cloud mask function for Sentinel-2 L2A using QA60
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
    
  return image.updateMask(mask).divide(10000); // Apply scale factor
}

// Load Sentinel-2 SR collection, filter, and calculate NDVI
var s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(samplePoints)
  .filterDate(startDate, endDate)
  .map(maskS2clouds)
  .map(function(img) {
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    return img.addBands(ndvi);
  });

// ==========================================
// 3. Monthly Median Compositing
// ==========================================
// Generate monthly median NDVI images (Month 1 to 12)
var months = ee.List.sequence(1, 12);
var monthlyNDVICol = ee.ImageCollection.fromImages(
  months.map(function (m) {
    var monthlyImg = s2Col.filter(ee.Filter.calendarRange(m, m, 'month'))
                          .select('NDVI')
                          .median();
    // Rename band to NDVI_1, NDVI_2, etc.
    var bandName = ee.String('NDVI_').cat(ee.Number(m).format('%d'));
    return monthlyImg.rename(bandName);
  })
);

// Stack the 12 monthly images into a single 12-band image
var monthlyStack = monthlyNDVICol.toBands();

// Clean up band names (remove the collection index prefix)
var bandNames = monthlyStack.bandNames();
var cleanBandNames = bandNames.map(function(name) {
  return ee.String(name).split('_').slice(1).join('_'); 
});
monthlyStack = monthlyStack.rename(cleanBandNames);

// ==========================================
// 4. Point Extraction and Export
// ==========================================
// Extract the 12-band NDVI values at each sample point location
var extractedData = monthlyStack.reduceRegions({
  collection: samplePoints,
  reducer: ee.Reducer.first(), 
  scale: 10,
  crs: 'EPSG:4326'
});

// Export the result to Google Drive as a CSV file
Export.table.toDrive({
  collection: extractedData,
  description: 'gee_raw_samples_ndvi', // Task name in GEE
  fileFormat: 'CSV',
  // Only export necessary columns to keep the CSV clean
  selectors: ['Species', 'NDVI_1', 'NDVI_2', 'NDVI_3', 'NDVI_4', 'NDVI_5', 
              'NDVI_6', 'NDVI_7', 'NDVI_8', 'NDVI_9', 'NDVI_10', 'NDVI_11', 'NDVI_12']
});
