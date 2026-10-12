# Maps, weather, saved models

## Building the map

::: areafluforecast.adjacency
    options:
      show_root_heading: false
      members: [adjacency_from_polygons, centroids_from_polygons, read_geojson, read_edges_csv, isolated]

## Weather

::: areafluforecast.weather
    options:
      show_root_heading: false
      members: [fetch_weather, read_weather_csv, WEATHER_COLS]

## Saved and pretrained models

::: areafluforecast.checkpoint.Checkpoint

::: areafluforecast.checkpoint.pretrained
