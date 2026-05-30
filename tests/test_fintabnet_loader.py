"""PASCAL VOC structure-XML parser test (CPU, synthetic).

Locks the parser contract. If the real archive uses different class strings, the
inspect run surfaces them and both the constants in fintabnet_loader and this fixture
are updated together.
"""

from src import fintabnet_loader as fl

_XML = """<annotation>
  <filename>tbl_0001.png</filename>
  <object><name>table</name>
    <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object>
  <object><name>table row</name>
    <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>25</ymax></bndbox></object>
  <object><name>table row</name>
    <bndbox><xmin>0</xmin><ymin>25</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object>
  <object><name>table column</name>
    <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
  <object><name>table column</name>
    <bndbox><xmin>50</xmin><ymin>0</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object>
  <object><name>table spanning cell</name>
    <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>25</ymax></bndbox></object>
  <object><name>table column header</name>
    <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>25</ymax></bndbox></object>
</annotation>"""


def test_parse_structure_xml(tmp_path):
    xml_path = tmp_path / "tbl_0001.xml"
    xml_path.write_text(_XML, encoding="utf-8")

    parsed = fl.parse_structure_xml(xml_path)

    assert parsed["image_filename"] == "tbl_0001.png"
    assert parsed["table_bbox"] == [0, 0, 100, 50]
    assert len(parsed["row_boxes"]) == 2
    assert len(parsed["col_boxes"]) == 2
    assert len(parsed["spanning_cells"]) == 1
    assert len(parsed["column_headers"]) == 1
    assert parsed["row_boxes"][0]["bbox"] == [0, 0, 100, 25]
    assert parsed["class_counts"]["table row"] == 2


def test_parsed_prediction_feeds_pipeline(tmp_path):
    # The parsed dict is shaped for normalize_tatr_prediction().
    from src.tatr_postprocess import normalize_tatr_prediction

    xml_path = tmp_path / "t.xml"
    xml_path.write_text(_XML, encoding="utf-8")
    parsed = fl.parse_structure_xml(xml_path)

    table = normalize_tatr_prediction(parsed)
    assert table["num_rows"] == 2
    assert table["num_cols"] == 2
    # 2x2 grid minus the 2 cells covered by the spanning cell + 1 merged = 3.
    assert len(table["cells"]) == 3
