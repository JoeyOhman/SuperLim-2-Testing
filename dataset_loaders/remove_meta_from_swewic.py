import json


def main():
    for split in ["train", "dev", "test"]:
        file_path = f"/home/joey/code/ai/SuperLim-2-Testing/data/SweWiC/{split}.jsonl"
        with open(file_path, 'r') as f:
            lines = f.readlines()

        objs = []
        for line in lines:
            json_obj = json.loads(line)
            # print(json_obj)
            # meta_dict = json_obj["meta"]
            if "meta" in json_obj:
                del json_obj["meta"]
            objs.append(json_obj)
            # print(meta_dict["first_saldo_sense_id"])
            # print(json_obj["meta"]["first_source"])
            # assert isinstance(meta_dict["first_source"], str)
            # assert isinstance(meta_dict["first_saldo_sense_id"], str)
            # assert isinstance(meta_dict["second_source"], str)
            # assert isinstance(meta_dict["second_saldo_sense_id"], str)
            # assert isinstance(meta_dict["lemma"], str)
            # assert isinstance(meta_dict["saldo_pos"], str)

        with open(file_path, 'w') as f:
            for obj in objs:
                out_str = json.dumps(obj, ensure_ascii=False)
                f.write(out_str + "\n")


if __name__ == '__main__':
    main()
