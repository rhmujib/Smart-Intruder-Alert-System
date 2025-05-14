import cv2
import datetime


def draw_overlays(frame, detection_timer_start, current_time, fps, patience_remaining):
    """Draw overlays on the frame such as FPS, datetime, person status, patience timer, and exit button"""
    # Draw FPS in the top-left corner
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw current date and time in the top-left corner with white color
    cv2.putText(frame, f"Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Draw patience time if it's set
    if patience_remaining is not None:
        cv2.putText(frame, f"Patience: {int(patience_remaining)}", (frame.shape[1] - 200, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


    return frame
