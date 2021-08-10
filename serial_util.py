def strings_share_characters(str1: str, str2: str) -> bool:
    """Determine if two strings share any characters."""
    for i in str2:
        if i in str1:
            return True

    return False


def get_numbers_from_text(text, separator="\t"):
    """Get a list of number from a string of numbers separated by :separator:[default: "\t"]."""
    try:
        if isinstance(text, bytearray) or isinstance(text, bytes):
            text = text.decode("utf-8")

        if (
                strings_share_characters(
                    text.lower(), "qwrtyuiopsasdfghjklzxcvbnm><*[]{}()"
                )
                or len(text) == 0
        ):
            return []

        return [float(i) for i in text.split(separator)]

    except Exception as e:
        print(f"get_numbers_from_text: {repr(e)} {repr(text)}")

        return []
