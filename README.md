# flask_parking
A web application that lets the user mark and monitor occupancy of parking spots.

Follow these steps to install the web application:
1. Install the Detectron2 application in the main folder.
2. Run 'pip install -r requirements.txt' to download the dependencies.
3. Download the Detectron2 trained model from this link: "https://drive.google.com/file/d/17CGd6k66RZ1TS6Vfot27Qtwl1rE8vldv/view?usp=share_link". Copy this file to the "detectron2_parking_model" folder.

Follow these steps to run the web application:
1. Navigate to the directory one step above the root folder "flask_parking".
2. Set the Flask application variable: "export FLASK_APP=flask_parking".
3. Run the application: "flask run".
