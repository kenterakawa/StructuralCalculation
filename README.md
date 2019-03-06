# StructuralCalculation
構造計算に使用するコード一式です。

## Composites
主に複合材の計算に使用するコードです。

#### CompositeCalculation
任意のCFRPレイアップの強度、剛性を計算します。

## PressureVessels
圧力容器、タンクの計算に使用するコードです。
ASME BPVC, ASME VIII Div.II設計ベース。

## PropellantTank
圧力容器ですが、座屈圧縮荷重を受けるもの、構造部材としての円筒の計算に使用するコードです。
胴体や、ストリンガ構造に使用します。

#### semimonocoque_structure
縦通材を有する円筒の座屈強度をBruhn式より求めます。
縦通材とスキンそれぞれの安全率を設定することで、必要ウェブ高さが求まります。

#### tanksizing
推進剤密度とターゲットO/Fからタンク容積と重量を求めます

#### tankstress
フライト中のタンクにかかる応力を求めます。

#### thickness
モノコック円筒タンクの座屈強度をBruhn式より求めます。

## SteelStructures
鉄骨構造の柱梁計算に使用するコードです。
