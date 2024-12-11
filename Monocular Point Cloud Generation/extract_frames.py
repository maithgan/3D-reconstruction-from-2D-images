import cv2
vidcap = cv2.VideoCapture(r"C:\Users\kaush\anaconda3\CVTerm\Stereo SFM\input\kaushek\vid.mp4")
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  cv2.imwrite(r"C:\Users\kaush\anaconda3\CVTerm\Stereo SFM\input\kaushek\frames\00%d.jpg" % count, image)     # save frame as JPEG file
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1