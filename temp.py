from minio import Minio
from minio.error import ResponseError
import glob
import tarfile
import shutil

minioClient = Minio('s3nverchev.ugent.be',
                  access_key='M12AX6PRAW2HUPUHPTL0',
                  secret_key='yC2UlxBD+exlGz5S+zLaVclYUvRQ9D8msgOMVAWh',secure=True)

with open("err.txt",'w') as file:
    file.write(str(0))

for subject in sorted(glob.glob("../data_vo/*")):
    print(subject)
    for operator in sorted(glob.glob(subject+'/*')):
        with open("err.txt", 'w') as file:
            file.write(operator)
        tarname=operator.split('/')[-1]+'_'+str(subject[-2:])+".tar.gz"
        tar = tarfile.open(tarname, "w:gz")
        tar.add(operator)
        tar.close()
        with open("err.txt", 'w') as file:
            file.write(str(1))
        minioClient.fput_object('coma', tarname, tarname)
        with open("err.txt", 'w') as file:
            file.write(str(2))
        shutil.rmtree(tarname)
        with open("err.txt", 'w') as file:
            file.write(str(3))
        print(operator)
        break




