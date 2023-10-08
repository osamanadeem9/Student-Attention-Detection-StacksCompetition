# VIGILANCE:
## Our Student attention detection module

Based on the problem statement "How to create an online classroom experience where student behaviour can be monitored in real-time?". It's an initial prototype, with a vision of high scalability. Submitted to StacksHack4Impact, and <b>were announced Runner-ups (2nd Position) at the hackathon</b>. <a href="https://www.linkedin.com/posts/stacks-pakistan_stackshack4impact-activity-6783103056239304704-RNpA">Link</a>

## High Level Overview
  * Uses Face and Gaze detector to determine Student availability/attentiveness.
  * Deep learning models are used for the facial emotional analysis of the student in realtime/from video.
  * Generates a raw report on frame by frame analysis
  
## Using the app

The application/project can be run on any Windows/Linux running device using python. Make sure to follow the steps below:
  * Firstly install the dependencies required to run the project. Run the command "pip install keras opencv-python imutils numpy" in command prompt.
  * Then using Spyder or PyCharm, run the "live_video.py" file, or using cmd, run the command "python live_video.py" to run the python file.
  * You may change the variable "use_live_video=True" to run the project on realtime webcam video, or provide a recorded video file to the model to process for the attentiveness/emotional analysis.

## Output

The working demo of Vigilance can be seen <a href = "https://drive.google.com/file/d/1bSe1jPE_ahDKvxGTjymriboka7_29E4d/view?usp=drive_link">here</a>:
