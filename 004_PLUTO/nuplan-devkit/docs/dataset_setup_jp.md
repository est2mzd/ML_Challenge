# データセットセットアップ

## ダウンロード

nuPlanをダウンロードするには、[ダウンロードページ](https://www.nuscenes.org/nuplan#download)にアクセスし、アカウントを作成して[利用規約](https://www.nuscenes.org/terms-of-use)に同意する必要があります。  
ログイン後、複数のアーカイブファイルが表示されます。開発キットを動作させるためには、**すべてのアーカイブ**をダウンロードする必要があります。  
ダウンロードしたアーカイブは `~/nuplan/dataset` フォルダに解凍してください。

## ファイルシステム階層

解凍後のフォルダ構成は以下のようになります（ダウンロードした分割データによって異なります）。

```bash
~/nuplan
├── exp
│   └── ${USER}
│       ├── cache
│       │   └── <cached_tokens>
│       └── exp
│           └── my_nuplan_experiment
└── dataset
    ├── maps
    │   ├── nuplan-maps-v1.0.json
    │   ├── sg-one-north
    │   │   └── 9.17.1964
    │   │       └── map.gpkg
    │   ├── us-ma-boston
    │   │   └── 9.12.1817
    │   │       └── map.gpkg
    │   ├── us-nv-las-vegas-strip
    │   │   └── 9.15.1915
    │   │       └── map.gpkg
    │   └── us-pa-pittsburgh-hazelwood
    │       └── 9.17.1937
    │           └── map.gpkg
    └── nuplan-v1.1
        ├── splits 
        │     ├── mini 
        │     │    ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
        │     │    ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
        │     │    ├── ...
        │     │    └── 2021.10.11.08.31.07_veh-50_01750_01948.db
        │     └── trainval
        │          ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
        │          ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
        │          ├── ...
        │          └── 2021.10.11.08.31.07_veh-50_01750_01948.db
        └── sensor_blobs   
              ├── 2021.05.12.22.00.38_veh-35_01008_01518                                           
              │    ├── CAM_F0
              │    │     ├── c082c104b7ac5a71.jpg
              │    │     ├── af380db4b4ca5d63.jpg
              │    │     ├── ...
              │    │     └── 2270fccfb44858b3.jpg
              │    ├── CAM_B0
              │    ├── CAM_L0
              │    ├── CAM_L1
              │    ├── CAM_L2
              │    ├── CAM_R0
              │    ├── CAM_R1
              │    ├── CAM_R2
              │    └──MergedPointCloud 
              │         ├── 03fafcf2c0865668.pcd  
              │         ├── 5aee37ce29665f1b.pcd  
              │         ├── ...                   
              │         └── 5fe65ef6a97f5caf.pcd  
              │
              ├── 2021.06.09.17.23.18_veh-38_00773_01140 
              ├── ...                                                                            
              └── 2021.10.11.08.31.07_veh-50_01750_01948
```

## カスタムフォルダの利用
別のフォルダを使用したい場合は、対応する[環境変数](https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md)を設定することで指定できます。