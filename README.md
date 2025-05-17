# Arabic Letter Prediction API

API for predicting Arabic letter types from images using TensorFlow deep learning model.

## Description

This application is a REST API that can be used to identify Arabic (Hijaiyah) letter types from submitted images. The API uses a machine learning model trained to recognize 30 Arabic letter characters with image input format.

Key features:
- Accepts images in various formats (JSON base64, form-data, raw binary)
- Displays letter predictions with confidence level
- Records prediction history and image samples for analysis
- Supports various use cases, including mobile application integration

## Installation

1. Clone this repository
```bash
git clone https://github.com/mhd-aris/flask-api-arabic-letter-prediction.git
cd flask-api-arabic-letter-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create directories for model and logs
```bash
mkdir -p model logs/samples
```

4. Place your machine learning model in the `model/` folder
```
model/model2.keras
```

5. Run the application
```bash
python app.py
```

## API Usage

### Endpoints

- `GET /` - Home page
- `GET /api-docs` - API documentation
- `POST /predict` - Endpoint for sending images and getting predictions

### Supported Request Formats

This API supports several request formats:

1. **JSON with Base64**
```json
{
  "file": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/..."
}
```

2. **Form data with 'file' field**
```
POST /predict
Content-Type: multipart/form-data

file: [image file]
```

3. **Form URL-encoded with Base64**
```
POST /predict
Content-Type: application/x-www-form-urlencoded

file=data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/...
```

4. **Raw Binary**
```
POST /predict
Content-Type: image/jpeg

[binary image data]
```

### Example Response

```json
{
  "predicted_class": "alif",
  "confidence": 0.9853621,
  "confidence_percent": "98.54%"
}
```

## MIT App Inventor Integration

### Using PostFile (Recommended Method)

Using `Web.PostFile` is the most reliable method for sending images from MIT App Inventor:

1. Add a `Web` component to your app
2. Set up the request as follows:
   - URL: `http://your-server-address:5000/predict`
   - Use the `PostFile` method with the path to your image
   
```
# Block Example
when Button1.Click
  call Web1.PostFile(
    Url: "http://your-server-address:5000/predict",
    Path: ImagePath
  )
```

### Using PostText (Alternative Method)

If you prefer to use `PostText`, follow these steps:

1. Take a picture or select an image
2. Draw the image to a Canvas
3. Convert the canvas to a data representation
4. Send using PostText with content-type "text/plain"

```
# Block Example
when ButtonSendImage.Click
  call Web1.PostText(
    Url: "http://your-server-address:5000/predict",
    Text: Canvas1.toBase64(),
    Encoding: "UTF-8",
    ContentType: "text/plain"
  )
```

### Common Issues and Solutions

1. **Empty Request Error**: 
   - Make sure you're specifying the correct path to the image file
   - Verify that the image exists before sending
   - Check that you're setting the proper URL

2. **Connection Issues**:
   - Make sure the server is running and accessible from your device
   - Check network permissions in your app
   - Try using a direct IP address instead of a hostname

3. **Debugging Tips**:
   - Add labels to display the response text
   - Use the debugging view in MIT App Inventor
   - Check the API logs at `/admin/logs?key=admin123` to see detailed request information

## Admin Endpoints

This API also provides several admin endpoints for monitoring:

- `GET /admin/logs?key=admin123` - View API logs
- `GET /admin/detailed-logs?key=admin123` - View detailed logs
- `GET /admin/samples?key=admin123` - View predicted image samples

## Security Notes

For production use, make sure to:
1. Replace the admin key `admin123` with a more secure value
2. Enable HTTPS
3. Set limiters to prevent DoS attacks
4. Disable debug mode

