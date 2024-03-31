import os
from fastavro import writer, parse_schema
import orjson as json
import data.schema.schema as schema_book
import inquirer
from loguru import logger

logger.add("avro_converter.log", format="{time} {level} {message}", level="INFO")


def get_file_size(file_path):
    return os.path.getsize(file_path)


class ARVOConverter:
    def __init__(
        self,
        schema: str,
        output_file_name: str,
        input_files: list[str],
    ):
        logger.info("Initializing ARVOConverter")
        logger.info(f"Schema: {schema}")
        logger.info(f"Output file name: {output_file_name}")
        logger.info(f"Input files: {input_files}")

        self.schema = getattr(schema_book, schema, None)
        self.output_file_name = output_file_name
        self.input_files = input_files

        if self.schema is None:
            logger.error(f"Invalid schema: {schema}")
            raise ValueError("Invalid schema")

    def _parse_json(self, json_path: str) -> dict:
        with open(json_path, "r") as f:
            data = json.loads(f.read())

        return data

    def start(self):
        data = []
        before_size = 0

        for file in self.input_files:
            logger.info(f"Parsing {file}")
            data.extend(self._parse_json(file))
            before_size += get_file_size(file)

        logger.info(f"변환 전 파일 크기: {before_size} 바이트")

        parsed_schema = parse_schema(self.schema)

        with open(self.output_file_name, "wb") as out:
            logger.info(f"Writing to {self.output_file_name}")
            writer(
                out,
                parsed_schema,
                data,
                validator=True,
                strict=True,
                strict_allow_default=True,
                disable_tuple_notation=True,
            )

        after_size = get_file_size(self.output_file_name)
        logger.info(f"변환 후 파일 크기: {after_size} 바이트")

        logger.info(f"변환률: {after_size / before_size * 100:.2f}%")

        logger.info("Conversion complete")


if __name__ == "__main__":
    questions = [
        inquirer.Text(
            "file_path",
            message="변환할 파일의 경로를 입력해주세요",
        ),
    ]

    file_path = inquirer.prompt(questions)["file_path"]

    if os.path.exists(file_path) is False:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError("File not found")

    file_list = os.listdir(file_path)
    file_list = [file for file in file_list if file.endswith(".json")]

    if len(file_list) == 0:
        logger.error(f"No JSON file found in {file_path}")
        raise FileNotFoundError("No JSON file found")

    questions = [
        inquirer.Checkbox(
            "input_file_names",
            message="변환할 파일을 선택해주세요",
            choices=file_list,
        ),
        inquirer.List(
            "schema",
            message="Schema를 선택해주세요",
            choices=["USER_LIST", "USER_MATCH_LIST"],
        ),
        inquirer.Text(
            "output_file_name",
            message="저장할 파일 이름을 선택해주세요",
            default="{schema}.avro",
        ),
    ]
    answers = inquirer.prompt(questions)

    avro_converter = ARVOConverter(
        answers["schema"],
        answers["output_file_name"],
        answers["input_file_names"],
    )
    avro_converter.start()
