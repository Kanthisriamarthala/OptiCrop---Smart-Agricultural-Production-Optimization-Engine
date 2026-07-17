import re


def validate_numeric_value(value, field_name, min_val=None, max_val=None):
    errors = []
    if not value or not str(value).strip():
        errors.append(f"{field_name} is required.")
        return errors
    try:
        val = float(value)
    except (ValueError, TypeError):
        errors.append(f"{field_name} must be a valid number.")
        return errors
    if min_val is not None and val < min_val:
        errors.append(f"{field_name} must be at least {min_val}.")
    if max_val is not None and val > max_val:
        errors.append(f"{field_name} must not exceed {max_val}.")
    return errors


def validate_prediction_input(data):
    fields = [
        ("nitrogen", "Nitrogen", 0, 300),
        ("phosphorous", "Phosphorous", 0, 300),
        ("potassium", "Potassium", 0, 500),
        ("temperature", "Temperature", -10, 60),
        ("humidity", "Humidity", 0, 100),
        ("ph", "pH", 0, 14),
        ("rainfall", "Rainfall", 0, 500),
    ]
    all_errors = {}
    for key, label, min_v, max_v in fields:
        errors = validate_numeric_value(data.get(key), label, min_v, max_v)
        if errors:
            all_errors[key] = errors
    return all_errors


def sanitize_input(value):
    if not isinstance(value, str):
        return value
    return re.sub(r"[<>&\"']", "", value.strip())
