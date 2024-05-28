import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from matplotlib.colors import LogNorm
from astropy.visualization import ZScaleInterval, ImageNormalize, SqrtStretch, LogStretch, MinMaxInterval, HistEqStretch
from io import BytesIO
from astropy.io import fits
from copy import deepcopy
import os





field = 856
makesize = '100x100'
path = 'images/'+makesize+'/'+str(field)+'/'
if not os.path.exists('images/'+makesize+'/'+str(field)): os.mkdir('images/'+makesize+'/'+str(field))

### Download fits with new header ###

def download_fits_by_url(url, ra, dec, oid, path, shape=(100,100)):
    new_url = url.replace('https://fits.ztf.snad.space/products/sci/', 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/')
    url_cuted_fits = new_url + f'?center={ra},{dec}&size={shape[0]-1}pix&gzip=false'
    
    response = requests.get(url_cuted_fits)
    response.raise_for_status()
    stream = BytesIO(response.content)
    stream.seek(0)
    hdus = fits.open(stream)
        
    tags = get_tags_by_oid(oid)

    ext_header = deepcopy(hdus[0].header)
    ext_header.append(('OID', oid, 'Object ID from SNAD Knowledge Database'), end=True)
    ext_header.append(('OIDRA', ra, 'Object RA (deg)'), end=True)
    ext_header.append(('OIDDEC', dec, 'Object DEC (deg)'), end=True)
    ext_header.append(('TAGS', str(tags), 'Object tag'), end=True)
    ext_header.append(('URL', url_cuted_fits, 'Download url'), end=True)
    hdu_newheader = fits.PrimaryHDU(deepcopy(hdus[0].data), ext_header)
    #
    hdu_newheader.writeto(path + str(oid) + '.fits', overwrite=True)
    
    hdus.close()





def get_tags_by_oid(oid):
    base_url = 'https://akb.ztf.snad.space/objects/'  # URL SNAD akb
    url = f'{base_url}{oid}'  
    
    response = requests.get(url)
    
    # Success of request
    if response.status_code == 200:
        data = response.json()
        tags = data.get('tags', [])
        return tags
    else:
        print(f"Failed to retrieve data for OID {oid}. HTTP Status code: {response.status_code}")
        return []



### Display images with tags ###

def display_fits_image(oid, path):
    fits_path = path + str(oid) + '.fits'
    hdus = fits.open(fits_path)
    
    data = hdus[0].data
    # Image normalization
    # norm = ImageNormalize(data, interval=ZScaleInterval())
    # norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=SqrtStretch())
    z = ZScaleInterval()
    z1,z2 = z.get_limits(data)
    norm = ImageNormalize(data, interval= MinMaxInterval(), stretch=HistEqStretch(data))
    #norm = normalize(data)
    # Extract tags for OID
    tags = get_tags_by_oid(oid)
    
    fig, ax = plt.subplots()
    ax.imshow(data, cmap='gray', origin='lower',norm=norm)
    
    title = f'OID: {oid}\nTags: {", ".join(tags)}'
    ax.set_title(title)
    plt.axis('off')
    if not os.path.exists('images/100x100/'+str(field)+'_png'): os.mkdir('images/100x100/'+str(field)+'_png')
    
    plt.savefig('images/100x100/'+str(field)+'_png/'+str(oid)+'.png',bbox_inches='tight')
    plt.close()
    hdus.close()
    if (len(tags)<2):
        print(oid)





df = pd.read_csv('data/Artefacts - '+str(field)+'.csv')





for index, row in df.iterrows():
    oid = str(row['oid'])
    url_fits = row['fits_image_link']
    url = 'http://db.ztf.snad.space/api/v3/data/latest/oid/meta/json?oid=' + oid
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        oid_data = data.get(oid)
        
        coord = oid_data.get('coord')
        ra = coord.get('ra')
        dec = coord.get('dec')
        download_fits_by_url(url_fits, ra, dec, oid, path)
        display_fits_image(oid, path)

