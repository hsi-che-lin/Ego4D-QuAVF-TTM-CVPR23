#!/bin/bash
if [ ! -d "video_imgs"  ];then
	mkdir "video_imgs"
fi
for file in `ls clips/*`
do
	name=$(basename $file .mp4)
	echo "$name"
	PTHH=video_imgs/$name
	if [ ! -d "$PTHH"  ];then
		mkdir "$PTHH"
	fi
	ffmpeg -i "$file" -f image2 -vf "fps=16,scale=-1:720" -qscale:v 2 "$PTHH/img_%05d.jpg"
done
