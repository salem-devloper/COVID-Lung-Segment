import os
import argparse
from tqdm import tqdm
import csv
from PIL import Image

def get_args():

    parser = argparse.ArgumentParser(description = "Qata_Covid19 Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path',type=str,default='./data/Qata_COV')
    parser.add_argument('--gpu', type=str, default = '0')
    # arguments for training
    parser.add_argument('--img_size', type = int , default = 224)

    parser.add_argument('--load_model', type=str, default='best_checkpoint.pt', help='.pth file path to load model')

    parser.add_argument('--out', type=str, default='./dataset')
    return parser.parse_args()

def main():
    args = get_args()
    
    if ~ os.path.exists(args.out):
        print("path created")
        os.mkdir(args.out)
        os.mkdir(os.path.join(args.out,'predict_crop_images_no'))
    
    img_path = os.path.join(args.path,'predict_crop_images/')
    img_list = os.listdir(img_path)
    croped_out = os.path.join(args.out,'predict_crop_images')
    i = 0

    with open(args.path+'/target.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img = Image.open(os.path.join(args.path,'predict_crop_images/'+img_list[i])).convert('L')
            if row['target'] == '0':
                Image.fromarray(img).save(os.path.join(croped_out,'no_'+str(row['img'])))
                #os.rename(r''+str(img_path)+row['img'],r''+str(img_path)+'no_'+row['img'])
            i = i + 1
#for i,img_name in tqdm(enumerate(img_list)):
#    os.rename(r''+str(img_path)+img_name,r''+str(img_path)+'no_'+img_name)

if __name__ == '__main__':

    main()

def process_image(img,kernels):

    # decode image ++

    footprint = np.array([[1,1,1],[1,1,1],[1,1,1]])

    decode_img = scipy.ndimage.generic_filter(img,convolve,footprint=footprint)

    decode_img = decode_img.reshape(-1)

    # calculate frequencies ++

    _, freqs = np.unique(decode_img, return_counts=True)

    freqs = np.sort(freqs)[::-1]

    # calculate rank ++
    
    rank = abs(rankdata(freqs,method='max')-freqs.shape[0]-1)  

    # calculate ferquencies >1 ++

    freqs_deleted_ones = np.delete(freqs,np.where(freqs == 1))

    # remove redandate frequencies
    
    unique_freqs,count_freqs = np.unique(freqs,return_counts=True)

    
    unique_freqs,count_freqs = unique_freqs[::-1],count_freqs[::-1]

    match = np.concatenate([np.expand_dims(unique_freqs,axis=1),np.expand_dims(count_freqs,axis=1)],axis=1)

    nbr_freqs=np.zeros(freqs.shape)

    for i in match:
        nbr_freqs[np.where(freqs == i[0])] = i[1]

    # entropy calculation

    p_e1 = freqs / np.sum(freqs)

    entropy_1 = -np.sum(p_e1 * np.log(p_e1))/math.log(freqs.shape[0])

    p_e2 = count_freqs / freqs.shape[0]

    entropy_2 = -np.sum(p_e2 * np.log(p_e2))/math.log(count_freqs.shape[0])

    # calculate slope

    u = np.log(freqs_deleted_ones)
    v= np.log(np.arange(1,freqs_deleted_ones.shape[0]+1))
    pente, constante= np.polyfit(u,v,1)

    # calculate air under zipf ++

    oao_zipf = math.log10(freqs[0]) 


    rank_deleted_ones = rank[:freqs_deleted_ones.shape[0]]
    
    air_zipf = np.sum((freqs_deleted_ones[:-1]+freqs_deleted_ones[1:])*(rank_deleted_ones[1:]-rank_deleted_ones[:-1])/2)

    # calculate zipf inverse
    
    u = np.log(freqs)
    v = np.log(nbr_freqs)

    zi_pente,_ = np.polyfit(u,v,1)

    oao_zipf_inv = math.log10(nbr_freqs[-1])

    # all zipf and zipf inverse features

    zipf_features = np.array([pente, constante, entropy_1, entropy_2, oao_zipf, air_zipf, oao_zipf_inv, zi_pente],dtype=np.float32)

    # calculate gabor features
    
    gabor_features_data = gabor_features(img,kernels,32,32)

    return np.concatenate([zipf_features, gabor_features_data])
