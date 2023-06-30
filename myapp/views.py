from django.http import JsonResponse
from django.http import HttpResponse
from scipy.fftpack import dct, idct
import numpy as np
from PIL import Image
import io

def set_boundaries(a: int):
    return max(min(a, 255), 0)
    
def process_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'Image file not found'})

        try:
            # Load image from file
            img = Image.open(image_file)

            # Convert image to numpy array
            imgmat = np.asarray(img)
            height = len(imgmat)
            width = len(imgmat[0])

            # Apply DCT to color channels
            red = dct(dct(imgmat[:,:,0],axis=0),axis=1)
            green = dct(dct(imgmat[:,:,1],axis=0),axis=1)
            blue = dct(dct(imgmat[:,:,2],axis=0),axis=1)

            # Modify DCT coefficients
            for i in range(height):
                for j in range(width):
                    if i > height / 4 or j > width / 4:
                        red[i][j] = set_boundaries(red[i][j] * 4 + 5)
                        green[i][j] = set_boundaries(2 * green[i][j])
                        blue[i][j] = set_boundaries(2 * blue[i][j])

            # Apply inverse DCT to modified coefficients
            imgmat2 = imgmat.copy()
            imgmat2[:,:,0] = idct(idct(red,axis=0),axis=1)/2/width/2/height
            imgmat2[:,:,1] = idct(idct(green,axis=0),axis=1)/2/width/2/height
            imgmat2[:,:,2] = idct(idct(blue,axis=0),axis=1)/2/width/2/height

            # Convert numpy array back to image
            img2 = Image.fromarray(imgmat2)
            # Convert image to RGB mode
            img2 = img2.convert('RGB')
            # Save output image to memory buffer
            output_buffer = io.BytesIO()
            img2.save(output_buffer, format='JPEG')
            output_buffer.seek(0)

            # Return output image as HTTP response
            return HttpResponse(content=output_buffer, content_type='image/jpeg')

        except Exception as e:
            return JsonResponse({'error': str(e)})
    else:
        return JsonResponse({'error': 'Invalid request method'})
