import constants as c
from flask import jsonify


def execute(request):
    if request.method == 'POST':
        input_data = request.get_json()
        if input_data:
            if c.input_text in input_data:
                string = input_data[c.input_text]
                output = f'Hello! {string}'
                output_data = {c.status: c.status_success,
                               c.title: "reverse string",
                               c.info: c.info_normal,
                               c.data: {c.message: output}}
                return jsonify(output_data)
            else:
                output_data = {c.status: c.status_failed,
                               c.title: "reverse string",
                               c.info: f"Expecting '{c.input_text}' as input!"}
                return jsonify(output_data)
        else:
            output_data = {c.status: c.status_failed,
                           c.title: "reverse string",
                           c.info: f"Expecting '{c.input_text}' as input in json format!"}
            return jsonify(output_data)
    else:
        output_data = {c.status: c.status_failed,
                       c.title: "reverse string",
                       c.info: f"Expecting POST method with '{c.input_text}' as input in json format!"}
        return jsonify(output_data)
