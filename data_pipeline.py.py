import os
import requests
import tarfile
import pandas as pd

#To extract the data from the internet 
data_dir = input("Enter path where dataset should be stored: ").strip().strip('"')
csv_dir = input("Enter path where CSV files should be stored: ").strip().strip('"')

data_dir = os.path.normpath(data_dir)
csv_dir = os.path.normpath(csv_dir)
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
tar_file = os.path.join(data_dir,"aclImdb_v1.tar.gz")

def download_file(url,filename,max_retries = 5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url,stream=True)
            total = int(response.headers.get('content-length',0))
            with open(filename,'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded+=len(chunk)
                    print(f"Download : {downloaded/1e6:.2f}/{total/1e6:.2f}")
                print("Downloaded Complete")
            return
        except:
            print(f"Download Failed.Trying again: {attempt+1}/{max_retries}")
            raise Exception(f"Download failed after {max_retries} times")
    
train_dir  = os.path.join(data_dir,'train')
if not os.path.exists(train_dir):
    print("File don't exists.Downloading...")
    download_file(url,tar_file)
    print("Extracting File")
    with tarfile.open(tar_file,'r:gz') as tar:
        tar.extractall(csv_dir)
    print("Extracted")
else:
    print("Data already exists")

#Converting all the text file present in the imdb dataset into a csv file
def load_reviews(folder_path,label):
    reviews = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path,file)
            with open (file_path,"r",encoding='utf-8') as f:
                text = f.read().strip()
                reviews.append([text,label])
    return reviews

train_pos_dir = load_reviews(os.path.join(data_dir,'train','pos'),1)
train_neg_dir = load_reviews(os.path.join(data_dir,'train',"neg"),0)

train = train_pos_dir+train_neg_dir
train_df = pd.DataFrame(train,columns=["reviews","sentiment"])

train_csv = os.path.join(csv_dir,"train.csv")
train_df.to_csv(train_csv,index=False)
print("Training data is prepared")

test_pos_dir = load_reviews(os.path.join(data_dir,'test','pos'),1)
test_neg_dir = load_reviews(os.path.join(data_dir,'test',"neg"),0)

test = test_pos_dir+test_neg_dir
test_df = pd.DataFrame(test,columns=["reviews","sentiment"])

test_csv = os.path.join(csv_dir,"test.csv")
test_df.to_csv(test_csv,index=False)
print("Testing data is prepared")