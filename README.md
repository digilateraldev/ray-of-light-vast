# ray-of-light   (DIGILATERAL)


Both Script perform same logic but on different template and background 

# Video processing steps

- Extract frames from video

- Apply rembg with human segmentation model

- Overlay that remove background image on custom background image    (part1.png)

- Apply ray of light by detecting the face on part1.png     (part2.png)
  
- Overlay custom template image on part2.png  (part3.png)

- Used part3.png image to make videos (normalvideo.mp4)

- Extract audio from input video

- Audio processing

- Add audio in normalvideo.mp4   (audiovideo.mp4)

- Add user information on audiovideo.mp4 like Name, Speciality and City  (Finalvideo.mp4)

- Save Finalvideo.mp4 (Outside the processed directory)

# Run Script


# 1) test2.py

a) without ray of light

- python test2.py inputvideo.mp4 output.mp4 "Name" "Specialist" "City"

b) with ray of light (soft ray)

- python test2.py inputvideo.mp4 output.mp4 "Name" "Specialist" "City" --intensity soft

c) with ray of light (hard ray)

- python test2.py inputvideo.mp4 output.mp4 "Name" "Specialist" "City" --intensity hard


# 2) try.py

a) without ray of light

- python try.py inputvideo.mp4 output.mp4 "Name" "Specialist" "City"

b) with ray of light (soft ray)

- python try.py inputvideo.mp4 output.mp4 "Name" "Specialist" "City" --intensity soft

c) with ray of light (hard ray)

- python try.py inputvideo.mp4 output.mp4 "Name" "Specialist" "City" --intensity hard
