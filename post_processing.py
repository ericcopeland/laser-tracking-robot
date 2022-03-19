import cv2


def add_tracking_field_lines(frame, data, **options):
    height = data['frame']['height']

    center_left_line = data['center']['left_line']
    center_right_line = data['center']['right_line']

    line_color = tuple(options['laser']['post_processing']['center_lines']['color'])
    line_thickness = options['laser']['post_processing']['center_lines']['thickness']

    cv2.line(
        frame,
        (center_left_line, 0),
        (center_left_line, height),
        line_color,
        line_thickness
    )
    cv2.line(
        frame,
        (center_right_line, 0),
        (center_right_line, height),
        line_color,
        line_thickness
    )

    return frame


def add_tracking_directions(frame, data, **options):
    x_pos = data['laser']['position']['left']
    y_pos = data['laser']['position']['top']

    center_left_line = data['center']['left_line']
    center_right_line = data['center']['right_line']

    line_color = tuple(options['laser']['post_processing']['control_arrow']['color'])
    line_thickness = options['laser']['post_processing']['control_arrow']['thickness']

    direction_text_options = options['laser']['post_processing']['direction_text']
    coordinates_options = options['laser']['post_processing']['coordinates']

    direction_text = None
    line_end = None

    if x_pos == 0 and y_pos == 0:
        line_end = (center_left_line, y_pos)
    elif x_pos < center_left_line:
        direction_text = 'MOVE RIGHT'
        line_end = (center_left_line, y_pos)
    elif x_pos > center_right_line:
        direction_text = 'MOVE LEFT'
        line_end = (center_right_line, y_pos)
    else:
        direction_text = 'CENTER'

    if direction_text != 'CENTER' and direction_text is not None:
        cv2.arrowedLine(frame, (x_pos, y_pos), line_end, line_color, line_thickness)

    cv2.putText(
        frame,
        direction_text,
        (x_pos, y_pos + direction_text_options['offset']),
        cv2.FONT_HERSHEY_SIMPLEX,
        direction_text_options['font_scale'],
        tuple(direction_text_options['color']),
        direction_text_options['thickness']
    )
    cv2.putText(
        frame,
        f'{(x_pos, y_pos)}',
        (x_pos, y_pos + coordinates_options['offset']),
        cv2.FONT_HERSHEY_SIMPLEX,
        coordinates_options['font_scale'],
        tuple(coordinates_options['color']),
        coordinates_options['thickness']
    )

    return frame


def add_landmine_indicators(frame, data, **options):
    text_options = options['landmine']['post_processing']['text']
    positions = data['landmine']['positions']

    for position in positions:
        x_pos = position['left']
        y_pos = position['top']
        cv2.putText(
            frame,
            text_options['content'],
            (x_pos, y_pos + text_options['offset']),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_options['font_scale'],
            tuple(text_options['color']),
            text_options['thickness']
        )

    return frame


def add_post_processing(frame, data, **options):
    frame = add_landmine_indicators(frame, data, **options)
    frame = add_tracking_field_lines(frame, data, **options)
    frame = add_tracking_directions(frame, data, **options)

    return frame
