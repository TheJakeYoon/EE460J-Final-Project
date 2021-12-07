import requests as requests
import urllib.request as url

for i in range(50):
    r = requests.get(f'https://api.unsplash.com/search/photos?query=cyberpunk&page={i}&per_page=50&client_id=pCvNl3DTJC_Qo54DuCkqrvHXO7WC8sPTCfTzOUfwF8Y')
    data = r.json()
    data.keys() # dict_keys(['total', 'total_pages', 'results'])
    data['total'] # 1393 images

    for img in data['results']:
        filename = "./unsplash/" + str(img['id']) + ".jpg"
        imgurl = img['urls']['small']
        url.urlretrieve(imgurl, filename)