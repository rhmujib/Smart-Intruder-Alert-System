from twilio.rest import Client
import cv2

# Twilio credentials
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
twilio_phone_number = 'your_twilio_number'
your_phone_number = 'your_phone_number'

twilio_client = Client(account_sid, auth_token)

def send_sms_alert():
    try:
        message = twilio_client.messages.create(
            body="Intruder detected! Check your laptop for more details.",
            from_=twilio_phone_number,
            to=your_phone_number
        )
        print(f"SMS sent: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

def init_video_writer(path, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(path, fourcc, 20.0, (width, height))
