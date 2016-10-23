ip=54.84.248.112
scp -i ~/.ssh/lucasbrunialti2.pem  ubuntu@$ip:~/Reconstruction.csv . &
scp -i ~/.ssh/lucasbrunialti2.pem  ubuntu@$ip:~/U.csv . &
scp -i ~/.ssh/lucasbrunialti2.pem  ubuntu@$ip:~/S.csv . &
scp -i ~/.ssh/lucasbrunialti2.pem  ubuntu@$ip:~/V.hdf5 . &
scp -i ~/.ssh/lucasbrunialti2.pem  ubuntu@$ip:~/error.csv . &

for job in `jobs -p`
do
echo $job
    wait $job
done

