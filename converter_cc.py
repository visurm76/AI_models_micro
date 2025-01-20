def convert_tflite_to_c_array(input_file, output_file):
    with open(input_file, "rb") as f:
        data = f.read()

    c_array = ", ".join(f"0x{byte:02x}" for byte in data)

    with open(output_file, "w") as f:
        f.write(f"const unsigned char model_tflite[] = {{{c_array}}};\n")
        f.write(f"const unsigned int model_tflite_len = {len(data)};\n")


# Пример использования
input_file = "venv/tflite/model.tflite"
output_file = "venv/tflite/model.cc"
convert_tflite_to_c_array(input_file, output_file)

