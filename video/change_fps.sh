#!/usr/bin/env bash

#for name in /home/lluc/Documents/ME16IN/devset/videos/*
#do
    # echo "$name"
    # cd $name/movies
    # cd $name
    # cd "/home/lluc/Documents/ME16IN/devset/videos/video_7"
    cd "/home/lluc/Documents/ME16IN/devset/251_videos"
    for file in *.mp4
    do
      echo "$file"
      # echo "$name/$file"
      # ffmpeg -i "$file" -qscale 0 -r 60 -y "$file"
      # ffmpeg -i "$file" -filter "minterpolate='fps=60'" "$name/$file"
      # ffmpeg -i "$file" -c copy -y -an "$file" # take audio
      ffmpeg -i "$file" -r 390 -an -filter "minterpolate='mi_mode=dup'" -y "dup_$file"
    done
    cd ..
#done
exit 0