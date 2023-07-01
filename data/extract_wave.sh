if [ ! -d "wave"  ];then
	mkdir "wave"
fi
for f in `ls clips/`
do
    echo ${f%.*}
    ffmpeg -y -i clips/${f} -qscale:a 0 -ac 1 -vn -threads 6 -ar 16000 wave/${f%.*}.wav -loglevel panic
done
