#ON REMOTE COMPUTER
#Get jupyter notebook up for remote access
source /student/sch923/Thesis/env/bin/activate
pip install -r requirements.txt
jupyter notebook --no-browser --port=8080

#ON LOCAL COMPUTER
#SSH into jupyter notebook
ssh -N -L 8080:localhost:8080 sch923@tuxworld.usask.ca