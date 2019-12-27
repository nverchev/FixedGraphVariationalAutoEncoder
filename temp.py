from minio import Minio
from minio.error import ResponseError
import glob
import tarfile
import shutil

minioClient = Minio('s3nverchev.ugent.be',
                  access_key='M12AX6PRAW2HUPUHPTL0',
                  secret_key='yC2UlxBD+exlGz5S+zLaVclYUvRQ9D8msgOMVAWh',secure=True)



for subject in sorted(glob.glob("../data_vo/*")):
    print(subject)
    for operator in sorted(glob.glob(subject)):
        tarname=operator.split('/')[-1]+'_'+str(subject[-2:-1])+".tar.gz"
        tar = tarfile.open(tarname, "w:gz")
        tar.add(operator)
        tar.close()
        minioClient.fput_object('coma', tarname, tarname)
        shutil.rmtree(tarname)
        print(operator)
        break




