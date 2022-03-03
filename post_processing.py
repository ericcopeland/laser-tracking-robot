import cv2


def add_tracking_field_lines(frame, center_width_percentage):
    height, width, _ = frame.shape

    center_width = center_width_percentage * width
    center_left_line = int((width / 2) - (center_width / 2))
    center_right_line = int((width / 2) + (center_width / 2))

    cv2.line(frame, (center_left_line, 0), (center_left_line, height), (255, 0, 255), 1)
    cv2.line(frame, (center_right_line, 0), (center_right_line, height), (255, 0, 255), 1)

    return frame


def add_tracking_directions(frame, max_loc, center_width_percentage):
    height, width, _ = frame.shape
    x_pos, y_pos = max_loc

    center_width = center_width_percentage * width
    center_left_line = int((width / 2) - (center_width / 2))
    center_right_line = int((width / 2) + (center_width / 2))

    direction_text = None
    line_end = None

    if x_pos == 0 and y_pos == 0:
        print('NO LASER DETECTED')
        line_end = (center_left_line, y_pos)
    elif x_pos < center_left_line:
        print('LASER ON LEFT SIDE')
        direction_text = 'MOVE RIGHT'
        line_end = (center_left_line, y_pos)
    elif x_pos > center_right_line:
        print('LASER ON RIGHT SIDE')
        direction_text = 'MOVE LEFT'
        line_end = (center_right_line, y_pos)
    else:
        direction_text = 'CENTER'
        print('LASER IN CENTER')

    if direction_text != 'CENTER' and direction_text is not None:
        cv2.arrowedLine(frame, max_loc, line_end, (0, 255, 255), 1)

    cv2.putText(frame, direction_text, (x_pos, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f'{max_loc}', (x_pos, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return frame


def add_post_processing(frame, max_loc):
    center_width_percentage = .1

    frame = add_tracking_field_lines(frame, center_width_percentage)
    frame = add_tracking_directions(frame, max_loc, center_width_percentage)

    return frame
