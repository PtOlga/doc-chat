---
tags:
- mteb
- sentence-transformers
- transformers
model-index:
- name: multilingual-e5-large-instruct
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (en)
      config: en
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 76.23880597014924
    - type: ap
      value: 39.07351965022687
    - type: f1
      value: 70.04836733862683
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (de)
      config: de
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 66.71306209850107
    - type: ap
      value: 79.01499914759529
    - type: f1
      value: 64.81951817560703
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (en-ext)
      config: en-ext
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 73.85307346326837
    - type: ap
      value: 22.447519885878737
    - type: f1
      value: 61.0162730745633
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (ja)
      config: ja
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 76.04925053533191
    - type: ap
      value: 23.44983217128922
    - type: f1
      value: 62.5723230907759
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_polarity
      name: MTEB AmazonPolarityClassification
      config: default
      split: test
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
    metrics:
    - type: accuracy
      value: 96.28742500000001
    - type: ap
      value: 94.8449918887462
    - type: f1
      value: 96.28680923610432
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (en)
      config: en
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 56.716
    - type: f1
      value: 55.76510398266401
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (de)
      config: de
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 52.99999999999999
    - type: f1
      value: 52.00829994765178
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (es)
      config: es
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 48.806000000000004
    - type: f1
      value: 48.082345914983634
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (fr)
      config: fr
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 48.507999999999996
    - type: f1
      value: 47.68752844642045
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (ja)
      config: ja
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 47.709999999999994
    - type: f1
      value: 47.05870376637181
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (zh)
      config: zh
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 44.662000000000006
    - type: f1
      value: 43.42371965372771
  - task:
      type: Retrieval
    dataset:
      type: arguana
      name: MTEB ArguAna
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 31.721
    - type: map_at_10
      value: 49.221
    - type: map_at_100
      value: 49.884
    - type: map_at_1000
      value: 49.888
    - type: map_at_3
      value: 44.31
    - type: map_at_5
      value: 47.276
    - type: mrr_at_1
      value: 32.432
    - type: mrr_at_10
      value: 49.5
    - type: mrr_at_100
      value: 50.163000000000004
    - type: mrr_at_1000
      value: 50.166
    - type: mrr_at_3
      value: 44.618
    - type: mrr_at_5
      value: 47.541
    - type: ndcg_at_1
      value: 31.721
    - type: ndcg_at_10
      value: 58.384
    - type: ndcg_at_100
      value: 61.111000000000004
    - type: ndcg_at_1000
      value: 61.187999999999995
    - type: ndcg_at_3
      value: 48.386
    - type: ndcg_at_5
      value: 53.708999999999996
    - type: precision_at_1
      value: 31.721
    - type: precision_at_10
      value: 8.741
    - type: precision_at_100
      value: 0.991
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 20.057
    - type: precision_at_5
      value: 14.609
    - type: recall_at_1
      value: 31.721
    - type: recall_at_10
      value: 87.411
    - type: recall_at_100
      value: 99.075
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 60.171
    - type: recall_at_5
      value: 73.044
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-p2p
      name: MTEB ArxivClusteringP2P
      config: default
      split: test
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
    metrics:
    - type: v_measure
      value: 46.40419580759799
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-s2s
      name: MTEB ArxivClusteringS2S
      config: default
      split: test
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
    metrics:
    - type: v_measure
      value: 40.48593255007969
  - task:
      type: Reranking
    dataset:
      type: mteb/askubuntudupquestions-reranking
      name: MTEB AskUbuntuDupQuestions
      config: default
      split: test
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
    metrics:
    - type: map
      value: 63.889179122289995
    - type: mrr
      value: 77.61146286769556
  - task:
      type: STS
    dataset:
      type: mteb/biosses-sts
      name: MTEB BIOSSES
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_pearson
      value: 88.15075203727929
    - type: cos_sim_spearman
      value: 86.9622224570873
    - type: euclidean_pearson
      value: 86.70473853624121
    - type: euclidean_spearman
      value: 86.9622224570873
    - type: manhattan_pearson
      value: 86.21089380980065
    - type: manhattan_spearman
      value: 86.75318154937008
  - task:
      type: BitextMining
    dataset:
      type: mteb/bucc-bitext-mining
      name: MTEB BUCC (de-en)
      config: de-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 99.65553235908142
    - type: f1
      value: 99.60681976339595
    - type: precision
      value: 99.58246346555325
    - type: recall
      value: 99.65553235908142
  - task:
      type: BitextMining
    dataset:
      type: mteb/bucc-bitext-mining
      name: MTEB BUCC (fr-en)
      config: fr-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 99.26260180497468
    - type: f1
      value: 99.14520507740848
    - type: precision
      value: 99.08650671362535
    - type: recall
      value: 99.26260180497468
  - task:
      type: BitextMining
    dataset:
      type: mteb/bucc-bitext-mining
      name: MTEB BUCC (ru-en)
      config: ru-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 98.07412538967787
    - type: f1
      value: 97.86629719431936
    - type: precision
      value: 97.76238309664012
    - type: recall
      value: 98.07412538967787
  - task:
      type: BitextMining
    dataset:
      type: mteb/bucc-bitext-mining
      name: MTEB BUCC (zh-en)
      config: zh-en
      split: test
      revision: d51519689f32196a32af33b075a01d0e7c51e252
    metrics:
    - type: accuracy
      value: 99.42074776197998
    - type: f1
      value: 99.38564156573635
    - type: precision
      value: 99.36808846761454
    - type: recall
      value: 99.42074776197998
  - task:
      type: Classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77Classification
      config: default
      split: test
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
    metrics:
    - type: accuracy
      value: 85.73376623376623
    - type: f1
      value: 85.68480707214599
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-p2p
      name: MTEB BiorxivClusteringP2P
      config: default
      split: test
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
    metrics:
    - type: v_measure
      value: 40.935218072113855
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-s2s
      name: MTEB BiorxivClusteringS2S
      config: default
      split: test
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
    metrics:
    - type: v_measure
      value: 36.276389017675264
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 27.764166666666668
    - type: map_at_10
      value: 37.298166666666674
    - type: map_at_100
      value: 38.530166666666666
    - type: map_at_1000
      value: 38.64416666666667
    - type: map_at_3
      value: 34.484833333333334
    - type: map_at_5
      value: 36.0385
    - type: mrr_at_1
      value: 32.93558333333333
    - type: mrr_at_10
      value: 41.589749999999995
    - type: mrr_at_100
      value: 42.425333333333334
    - type: mrr_at_1000
      value: 42.476333333333336
    - type: mrr_at_3
      value: 39.26825
    - type: mrr_at_5
      value: 40.567083333333336
    - type: ndcg_at_1
      value: 32.93558333333333
    - type: ndcg_at_10
      value: 42.706583333333334
    - type: ndcg_at_100
      value: 47.82483333333333
    - type: ndcg_at_1000
      value: 49.95733333333334
    - type: ndcg_at_3
      value: 38.064750000000004
    - type: ndcg_at_5
      value: 40.18158333333333
    - type: precision_at_1
      value: 32.93558333333333
    - type: precision_at_10
      value: 7.459833333333334
    - type: precision_at_100
      value: 1.1830833333333335
    - type: precision_at_1000
      value: 0.15608333333333332
    - type: precision_at_3
      value: 17.5235
    - type: precision_at_5
      value: 12.349833333333333
    - type: recall_at_1
      value: 27.764166666666668
    - type: recall_at_10
      value: 54.31775
    - type: recall_at_100
      value: 76.74350000000001
    - type: recall_at_1000
      value: 91.45208333333332
    - type: recall_at_3
      value: 41.23425
    - type: recall_at_5
      value: 46.73983333333334
  - task:
      type: Retrieval
    dataset:
      type: climate-fever
      name: MTEB ClimateFEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 12.969
    - type: map_at_10
      value: 21.584999999999997
    - type: map_at_100
      value: 23.3
    - type: map_at_1000
      value: 23.5
    - type: map_at_3
      value: 18.218999999999998
    - type: map_at_5
      value: 19.983
    - type: mrr_at_1
      value: 29.316
    - type: mrr_at_10
      value: 40.033
    - type: mrr_at_100
      value: 40.96
    - type: mrr_at_1000
      value: 41.001
    - type: mrr_at_3
      value: 37.123
    - type: mrr_at_5
      value: 38.757999999999996
    - type: ndcg_at_1
      value: 29.316
    - type: ndcg_at_10
      value: 29.858
    - type: ndcg_at_100
      value: 36.756
    - type: ndcg_at_1000
      value: 40.245999999999995
    - type: ndcg_at_3
      value: 24.822
    - type: ndcg_at_5
      value: 26.565
    - type: precision_at_1
      value: 29.316
    - type: precision_at_10
      value: 9.186
    - type: precision_at_100
      value: 1.6549999999999998
    - type: precision_at_1000
      value: 0.22999999999999998
    - type: precision_at_3
      value: 18.436
    - type: precision_at_5
      value: 13.876
    - type: recall_at_1
      value: 12.969
    - type: recall_at_10
      value: 35.142
    - type: recall_at_100
      value: 59.143
    - type: recall_at_1000
      value: 78.594
    - type: recall_at_3
      value: 22.604
    - type: recall_at_5
      value: 27.883000000000003
  - task:
      type: Retrieval
    dataset:
      type: dbpedia-entity
      name: MTEB DBPedia
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 8.527999999999999
    - type: map_at_10
      value: 17.974999999999998
    - type: map_at_100
      value: 25.665
    - type: map_at_1000
      value: 27.406000000000002
    - type: map_at_3
      value: 13.017999999999999
    - type: map_at_5
      value: 15.137
    - type: mrr_at_1
      value: 62.5
    - type: mrr_at_10
      value: 71.891
    - type: mrr_at_100
      value: 72.294
    - type: mrr_at_1000
      value: 72.296
    - type: mrr_at_3
      value: 69.958
    - type: mrr_at_5
      value: 71.121
    - type: ndcg_at_1
      value: 50.875
    - type: ndcg_at_10
      value: 38.36
    - type: ndcg_at_100
      value: 44.235
    - type: ndcg_at_1000
      value: 52.154
    - type: ndcg_at_3
      value: 43.008
    - type: ndcg_at_5
      value: 40.083999999999996
    - type: precision_at_1
      value: 62.5
    - type: precision_at_10
      value: 30.0
    - type: precision_at_100
      value: 10.038
    - type: precision_at_1000
      value: 2.0869999999999997
    - type: precision_at_3
      value: 46.833000000000006
    - type: precision_at_5
      value: 38.800000000000004
    - type: recall_at_1
      value: 8.527999999999999
    - type: recall_at_10
      value: 23.828
    - type: recall_at_100
      value: 52.322
    - type: recall_at_1000
      value: 77.143
    - type: recall_at_3
      value: 14.136000000000001
    - type: recall_at_5
      value: 17.761
  - task:
      type: Classification
    dataset:
      type: mteb/emotion
      name: MTEB EmotionClassification
      config: default
      split: test
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
    metrics:
    - type: accuracy
      value: 51.51
    - type: f1
      value: 47.632159862049896
  - task:
      type: Retrieval
    dataset:
      type: fever
      name: MTEB FEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 60.734
    - type: map_at_10
      value: 72.442
    - type: map_at_100
      value: 72.735
    - type: map_at_1000
      value: 72.75
    - type: map_at_3
      value: 70.41199999999999
    - type: map_at_5
      value: 71.80499999999999
    - type: mrr_at_1
      value: 65.212
    - type: mrr_at_10
      value: 76.613
    - type: mrr_at_100
      value: 76.79899999999999
    - type: mrr_at_1000
      value: 76.801
    - type: mrr_at_3
      value: 74.8
    - type: mrr_at_5
      value: 76.12400000000001
    - type: ndcg_at_1
      value: 65.212
    - type: ndcg_at_10
      value: 77.988
    - type: ndcg_at_100
      value: 79.167
    - type: ndcg_at_1000
      value: 79.452
    - type: ndcg_at_3
      value: 74.362
    - type: ndcg_at_5
      value: 76.666
    - type: precision_at_1
      value: 65.212
    - type: precision_at_10
      value: 10.003
    - type: precision_at_100
      value: 1.077
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 29.518
    - type: precision_at_5
      value: 19.016
    - type: recall_at_1
      value: 60.734
    - type: recall_at_10
      value: 90.824
    - type: recall_at_100
      value: 95.71600000000001
    - type: recall_at_1000
      value: 97.577
    - type: recall_at_3
      value: 81.243
    - type: recall_at_5
      value: 86.90299999999999
  - task:
      type: Retrieval
    dataset:
      type: fiqa
      name: MTEB FiQA2018
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 23.845
    - type: map_at_10
      value: 39.281
    - type: map_at_100
      value: 41.422
    - type: map_at_1000
      value: 41.593
    - type: map_at_3
      value: 34.467
    - type: map_at_5
      value: 37.017
    - type: mrr_at_1
      value: 47.531
    - type: mrr_at_10
      value: 56.204
    - type: mrr_at_100
      value: 56.928999999999995
    - type: mrr_at_1000
      value: 56.962999999999994
    - type: mrr_at_3
      value: 54.115
    - type: mrr_at_5
      value: 55.373000000000005
    - type: ndcg_at_1
      value: 47.531
    - type: ndcg_at_10
      value: 47.711999999999996
    - type: ndcg_at_100
      value: 54.510999999999996
    - type: ndcg_at_1000
      value: 57.103
    - type: ndcg_at_3
      value: 44.145
    - type: ndcg_at_5
      value: 45.032
    - type: precision_at_1
      value: 47.531
    - type: precision_at_10
      value: 13.194
    - type: precision_at_100
      value: 2.045
    - type: precision_at_1000
      value: 0.249
    - type: precision_at_3
      value: 29.424
    - type: precision_at_5
      value: 21.451
    - type: recall_at_1
      value: 23.845
    - type: recall_at_10
      value: 54.967
    - type: recall_at_100
      value: 79.11399999999999
    - type: recall_at_1000
      value: 94.56700000000001
    - type: recall_at_3
      value: 40.256
    - type: recall_at_5
      value: 46.215
  - task:
      type: Retrieval
    dataset:
      type: hotpotqa
      name: MTEB HotpotQA
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 37.819
    - type: map_at_10
      value: 60.889
    - type: map_at_100
      value: 61.717999999999996
    - type: map_at_1000
      value: 61.778
    - type: map_at_3
      value: 57.254000000000005
    - type: map_at_5
      value: 59.541
    - type: mrr_at_1
      value: 75.638
    - type: mrr_at_10
      value: 82.173
    - type: mrr_at_100
      value: 82.362
    - type: mrr_at_1000
      value: 82.37
    - type: mrr_at_3
      value: 81.089
    - type: mrr_at_5
      value: 81.827
    - type: ndcg_at_1
      value: 75.638
    - type: ndcg_at_10
      value: 69.317
    - type: ndcg_at_100
      value: 72.221
    - type: ndcg_at_1000
      value: 73.382
    - type: ndcg_at_3
      value: 64.14
    - type: ndcg_at_5
      value: 67.07600000000001
    - type: precision_at_1
      value: 75.638
    - type: precision_at_10
      value: 14.704999999999998
    - type: precision_at_100
      value: 1.698
    - type: precision_at_1000
      value: 0.185
    - type: precision_at_3
      value: 41.394999999999996
    - type: precision_at_5
      value: 27.162999999999997
    - type: recall_at_1
      value: 37.819
    - type: recall_at_10
      value: 73.52499999999999
    - type: recall_at_100
      value: 84.875
    - type: recall_at_1000
      value: 92.559
    - type: recall_at_3
      value: 62.092999999999996
    - type: recall_at_5
      value: 67.907
  - task:
      type: Classification
    dataset:
      type: mteb/imdb
      name: MTEB ImdbClassification
      config: default
      split: test
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
    metrics:
    - type: accuracy
      value: 94.60079999999999
    - type: ap
      value: 92.67396345347356
    - type: f1
      value: 94.5988098167121
  - task:
      type: Retrieval
    dataset:
      type: msmarco
      name: MTEB MSMARCO
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 21.285
    - type: map_at_10
      value: 33.436
    - type: map_at_100
      value: 34.63
    - type: map_at_1000
      value: 34.681
    - type: map_at_3
      value: 29.412
    - type: map_at_5
      value: 31.715
    - type: mrr_at_1
      value: 21.848
    - type: mrr_at_10
      value: 33.979
    - type: mrr_at_100
      value: 35.118
    - type: mrr_at_1000
      value: 35.162
    - type: mrr_at_3
      value: 30.036
    - type: mrr_at_5
      value: 32.298
    - type: ndcg_at_1
      value: 21.862000000000002
    - type: ndcg_at_10
      value: 40.43
    - type: ndcg_at_100
      value: 46.17
    - type: ndcg_at_1000
      value: 47.412
    - type: ndcg_at_3
      value: 32.221
    - type: ndcg_at_5
      value: 36.332
    - type: precision_at_1
      value: 21.862000000000002
    - type: precision_at_10
      value: 6.491
    - type: precision_at_100
      value: 0.935
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 13.744
    - type: precision_at_5
      value: 10.331999999999999
    - type: recall_at_1
      value: 21.285
    - type: recall_at_10
      value: 62.083
    - type: recall_at_100
      value: 88.576
    - type: recall_at_1000
      value: 98.006
    - type: recall_at_3
      value: 39.729
    - type: recall_at_5
      value: 49.608000000000004
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (en)
      config: en
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 93.92612859097127
    - type: f1
      value: 93.82370333372853
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (de)
      config: de
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 92.67681036911807
    - type: f1
      value: 92.14191382411472
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (es)
      config: es
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 92.26817878585723
    - type: f1
      value: 91.92824250337878
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (fr)
      config: fr
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 89.96554963983714
    - type: f1
      value: 90.02859329630792
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (hi)
      config: hi
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 90.02509860164935
    - type: f1
      value: 89.30665159182062
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (th)
      config: th
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 87.55515370705244
    - type: f1
      value: 87.94449232331907
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (en)
      config: en
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 82.4623803009576
    - type: f1
      value: 66.06738378772725
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (de)
      config: de
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 79.3716539870386
    - type: f1
      value: 60.37614033396853
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (es)
      config: es
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 80.34022681787857
    - type: f1
      value: 58.302008026952
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (fr)
      config: fr
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 76.72095208268087
    - type: f1
      value: 59.64524724009049
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (hi)
      config: hi
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 77.87020437432773
    - type: f1
      value: 57.80202694670567
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (th)
      config: th
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 77.73598553345387
    - type: f1
      value: 58.19628250675031
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (af)
      config: af
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 67.6630800268998
    - type: f1
      value: 65.00996668051691
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (am)
      config: am
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 60.7128446536651
    - type: f1
      value: 57.95860594874963
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (ar)
      config: ar
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 63.61129791526563
    - type: f1
      value: 59.75328290206483
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (az)
      config: az
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 69.00134498991257
    - type: f1
      value: 67.0230483991802
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (bn)
      config: bn
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 68.54068594485541
    - type: f1
      value: 65.54604628946976
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (cy)
      config: cy
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 63.032952252858095
    - type: f1
      value: 58.715741857057104
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (da)
      config: da
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 71.80901143241427
    - type: f1
      value: 68.33963989243877
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (de)
      config: de
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 72.47141896435777
    - type: f1
      value: 69.56765020308262
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (el)
      config: el
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 71.2373907195696
    - type: f1
      value: 69.04529836036467
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (en)
      config: en
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 77.05783456624076
    - type: f1
      value: 74.69430584708174
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (es)
      config: es
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 72.82111634162744
    - type: f1
      value: 70.77228952803762
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (fa)
      config: fa
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 74.25353059852051
    - type: f1
      value: 71.05310103416411
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (fi)
      config: fi
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 72.28648285137861
    - type: f1
      value: 69.08020473732226
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (fr)
      config: fr
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 73.31540013449899
    - type: f1
      value: 70.9426355465791
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (he)
      config: he
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 70.2151983860121
    - type: f1
      value: 67.52541755908858
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (hi)
      config: hi
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 71.58372562205784
    - type: f1
      value: 69.49769064229827
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (hu)
      config: hu
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 71.9233355749832
    - type: f1
      value: 69.36311548259593
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (hy)
      config: hy
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 68.07330195023538
    - type: f1
      value: 64.99882022345572
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (id)
      config: id
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 72.62273032952253
    - type: f1
      value: 70.6394885471001
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (is)
      config: is
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 65.77000672494957
    - type: f1
      value: 62.9368944815065
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (it)
      config: it
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 73.453261600538
    - type: f1
      value: 70.85069934666681
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (ja)
      config: ja
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 74.6906523201076
    - type: f1
      value: 72.03249740074217
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (jv)
      config: jv
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 63.03631472763953
    - type: f1
      value: 59.3165215571852
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (ka)
      config: ka
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 58.913920645595155
    - type: f1
      value: 57.367337711611285
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (km)
      config: km
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 54.42837928715535
    - type: f1
      value: 52.60527294970906
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (kn)
      config: kn
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 66.33490248823135
    - type: f1
      value: 63.213340969404065
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (ko)
      config: ko
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 70.58507061197041
    - type: f1
      value: 68.40256628040486
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (lv)
      config: lv
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 69.11230665770006
    - type: f1
      value: 66.44863577842305
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (ml)
      config: ml
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 69.70073974445192
    - type: f1
      value: 67.21291337273702
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (mn)
      config: mn
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 66.43913920645595
    - type: f1
      value: 64.09838087422806
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (ms)
      config: ms
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 70.80026899798251
    - type: f1
      value: 68.76986742962444
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (my)
      config: my
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 64.78816408876934
    - type: f1
      value: 62.18781873428972
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (nb)
      config: nb
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 71.6577000672495
    - type: f1
      value: 68.75171511133003
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (nl)
      config: nl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 74.42501681237391
    - type: f1
      value: 71.18434963451544
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (pl)
      config: pl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 73.64828513786146
    - type: f1
      value: 70.67741914007422
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (pt)
      config: pt
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 73.62811028917284
    - type: f1
      value: 71.36402039740959
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (ro)
      config: ro
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 71.88634835238736
    - type: f1
      value: 69.23701923480677
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (ru)
      config: ru
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 74.15938130464022
    - type: f1
      value: 71.87792218993388
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (sl)
      config: sl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 69.96301277740416
    - type: f1
      value: 67.29584200202983
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (sq)
      config: sq
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 69.49562878278412
    - type: f1
      value: 66.91716685679431
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (sv)
      config: sv
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 74.6805648957633
    - type: f1
      value: 72.02723592594374
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (sw)
      config: sw
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 63.00605245460659
    - type: f1
      value: 60.16716669482932
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (ta)
      config: ta
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 66.90988567585742
    - type: f1
      value: 63.99405488777784
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (te)
      config: te
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 67.62273032952253
    - type: f1
      value: 65.17213906909481
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (th)
      config: th
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 69.50907868190988
    - type: f1
      value: 69.15165697194853
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (tl)
      config: tl
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 69.30733019502352
    - type: f1
      value: 66.69024007380474
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (tr)
      config: tr
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 72.24277067921989
    - type: f1
      value: 68.80515408492947
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (ur)
      config: ur
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 67.49831876260929
    - type: f1
      value: 64.83778567111116
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (vi)
      config: vi
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 71.28782784129119
    - type: f1
      value: 69.3294186700733
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (zh-CN)
      config: zh-CN
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 73.315400134499
    - type: f1
      value: 71.22674385243207
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (zh-TW)
      config: zh-TW
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 69.37794216543377
    - type: f1
      value: 68.96962492838232
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (af)
      config: af
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 73.33557498318764
    - type: f1
      value: 72.28949738478356
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (am)
      config: am
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 65.84398117014123
    - type: f1
      value: 64.71026362091463
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (ar)
      config: ar
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 69.76462676529925
    - type: f1
      value: 69.8229667407667
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (az)
      config: az
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 72.02420981842636
    - type: f1
      value: 71.76576384895898
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (bn)
      config: bn
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 72.7572293207801
    - type: f1
      value: 72.76840765295256
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (cy)
      config: cy
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 68.02286482851379
    - type: f1
      value: 66.17237947327872
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (da)
      config: da
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 77.60928043039678
    - type: f1
      value: 77.27094731234773
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (de)
      config: de
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 77.68325487558843
    - type: f1
      value: 77.97530399082261
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (el)
      config: el
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 76.13315400134498
    - type: f1
      value: 75.97558584796424
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (en)
      config: en
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 80.47410894418292
    - type: f1
      value: 80.52244841473792
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (es)
      config: es
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 76.9670477471419
    - type: f1
      value: 77.37318805793146
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (fa)
      config: fa
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 78.09683927370544
    - type: f1
      value: 77.69773737430847
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (fi)
      config: fi
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 75.20847343644922
    - type: f1
      value: 75.17071738727348
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (fr)
      config: fr
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 77.07464694014796
    - type: f1
      value: 77.16136207698571
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (he)
      config: he
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 73.53396099529255
    - type: f1
      value: 73.58296404484122
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (hi)
      config: hi
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 75.75319435104237
    - type: f1
      value: 75.24674707850833
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (hu)
      config: hu
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 77.0948217888366
    - type: f1
      value: 76.47559490205028
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (hy)
      config: hy
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 71.07599193006052
    - type: f1
      value: 70.76028043093511
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (id)
      config: id
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 77.10490921318089
    - type: f1
      value: 77.01215275283272
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (is)
      config: is
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 71.25756556825824
    - type: f1
      value: 70.20605314648762
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (it)
      config: it
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 77.08137188971082
    - type: f1
      value: 77.3899269057439
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (ja)
      config: ja
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 79.35440484196369
    - type: f1
      value: 79.58964690002772
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (jv)
      config: jv
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 68.42299932750504
    - type: f1
      value: 68.07844356925413
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (ka)
      config: ka
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 66.15669132481507
    - type: f1
      value: 65.89383352608513
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (km)
      config: km
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 60.11432414256894
    - type: f1
      value: 57.69910594559806
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (kn)
      config: kn
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 71.24747814391392
    - type: f1
      value: 70.42455553830918
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (ko)
      config: ko
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 76.46267652992603
    - type: f1
      value: 76.8854559308316
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (lv)
      config: lv
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 73.24815063887021
    - type: f1
      value: 72.77805034658074
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (ml)
      config: ml
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 74.11566913248151
    - type: f1
      value: 73.86147988001356
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (mn)
      config: mn
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 70.0168123739072
    - type: f1
      value: 69.38515920054571
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (ms)
      config: ms
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 74.41156691324814
    - type: f1
      value: 73.43474953408237
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (my)
      config: my
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 68.39609952925353
    - type: f1
      value: 67.29731681109291
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (nb)
      config: nb
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 77.20914593140552
    - type: f1
      value: 77.07066497935367
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (nl)
      config: nl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 78.52387357094821
    - type: f1
      value: 78.5259569473291
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (pl)
      config: pl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 76.6913248150639
    - type: f1
      value: 76.91201656350455
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (pt)
      config: pt
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 77.1217215870881
    - type: f1
      value: 77.41179937912504
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (ro)
      config: ro
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 75.25891055817083
    - type: f1
      value: 75.8089244542887
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (ru)
      config: ru
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 77.70679219905851
    - type: f1
      value: 78.21459594517711
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (sl)
      config: sl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 74.83523873570948
    - type: f1
      value: 74.86847028401978
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (sq)
      config: sq
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 74.71755211835911
    - type: f1
      value: 74.0214326485662
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (sv)
      config: sv
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 79.06523201075991
    - type: f1
      value: 79.10545620325138
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (sw)
      config: sw
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 67.91862811028918
    - type: f1
      value: 66.50386121217983
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (ta)
      config: ta
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 70.93140551445865
    - type: f1
      value: 70.755435928495
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (te)
      config: te
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 72.40753194351042
    - type: f1
      value: 71.61816115782923
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (th)
      config: th
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 75.1815736381977
    - type: f1
      value: 75.08016717887205
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (tl)
      config: tl
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 72.86482851378614
    - type: f1
      value: 72.39521180006291
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (tr)
      config: tr
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 76.46940147948891
    - type: f1
      value: 76.70044085362349
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (ur)
      config: ur
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 71.89307330195024
    - type: f1
      value: 71.5721825332298
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (vi)
      config: vi
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 74.7511768661735
    - type: f1
      value: 75.17918654541515
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (zh-CN)
      config: zh-CN
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 78.69535978480162
    - type: f1
      value: 78.90019070153316
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (zh-TW)
      config: zh-TW
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 75.45729657027572
    - type: f1
      value: 76.19578371794672
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-p2p
      name: MTEB MedrxivClusteringP2P
      config: default
      split: test
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
    metrics:
    - type: v_measure
      value: 36.92715354123554
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-s2s
      name: MTEB MedrxivClusteringS2S
      config: default
      split: test
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
    metrics:
    - type: v_measure
      value: 35.53536244162518
  - task:
      type: Reranking
    dataset:
      type: mteb/mind_small
      name: MTEB MindSmallReranking
      config: default
      split: test
      revision: 3bdac13927fdc888b903db93b2ffdbd90b295a69
    metrics:
    - type: map
      value: 33.08507884504006
    - type: mrr
      value: 34.32436977159129
  - task:
      type: Retrieval
    dataset:
      type: nfcorpus
      name: MTEB NFCorpus
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 5.935
    - type: map_at_10
      value: 13.297
    - type: map_at_100
      value: 16.907
    - type: map_at_1000
      value: 18.391
    - type: map_at_3
      value: 9.626999999999999
    - type: map_at_5
      value: 11.190999999999999
    - type: mrr_at_1
      value: 46.129999999999995
    - type: mrr_at_10
      value: 54.346000000000004
    - type: mrr_at_100
      value: 55.067
    - type: mrr_at_1000
      value: 55.1
    - type: mrr_at_3
      value: 51.961
    - type: mrr_at_5
      value: 53.246
    - type: ndcg_at_1
      value: 44.118
    - type: ndcg_at_10
      value: 35.534
    - type: ndcg_at_100
      value: 32.946999999999996
    - type: ndcg_at_1000
      value: 41.599000000000004
    - type: ndcg_at_3
      value: 40.25
    - type: ndcg_at_5
      value: 37.978
    - type: precision_at_1
      value: 46.129999999999995
    - type: precision_at_10
      value: 26.842
    - type: precision_at_100
      value: 8.427
    - type: precision_at_1000
      value: 2.128
    - type: precision_at_3
      value: 37.977
    - type: precision_at_5
      value: 32.879000000000005
    - type: recall_at_1
      value: 5.935
    - type: recall_at_10
      value: 17.211000000000002
    - type: recall_at_100
      value: 34.33
    - type: recall_at_1000
      value: 65.551
    - type: recall_at_3
      value: 10.483
    - type: recall_at_5
      value: 13.078999999999999
  - task:
      type: Retrieval
    dataset:
      type: nq
      name: MTEB NQ
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 35.231
    - type: map_at_10
      value: 50.202000000000005
    - type: map_at_100
      value: 51.154999999999994
    - type: map_at_1000
      value: 51.181
    - type: map_at_3
      value: 45.774
    - type: map_at_5
      value: 48.522
    - type: mrr_at_1
      value: 39.687
    - type: mrr_at_10
      value: 52.88
    - type: mrr_at_100
      value: 53.569
    - type: mrr_at_1000
      value: 53.58500000000001
    - type: mrr_at_3
      value: 49.228
    - type: mrr_at_5
      value: 51.525
    - type: ndcg_at_1
      value: 39.687
    - type: ndcg_at_10
      value: 57.754000000000005
    - type: ndcg_at_100
      value: 61.597
    - type: ndcg_at_1000
      value: 62.18900000000001
    - type: ndcg_at_3
      value: 49.55
    - type: ndcg_at_5
      value: 54.11899999999999
    - type: precision_at_1
      value: 39.687
    - type: precision_at_10
      value: 9.313
    - type: precision_at_100
      value: 1.146
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 22.229
    - type: precision_at_5
      value: 15.939
    - type: recall_at_1
      value: 35.231
    - type: recall_at_10
      value: 78.083
    - type: recall_at_100
      value: 94.42099999999999
    - type: recall_at_1000
      value: 98.81
    - type: recall_at_3
      value: 57.047000000000004
    - type: recall_at_5
      value: 67.637
  - task:
      type: Retrieval
    dataset:
      type: quora
      name: MTEB QuoraRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 71.241
    - type: map_at_10
      value: 85.462
    - type: map_at_100
      value: 86.083
    - type: map_at_1000
      value: 86.09700000000001
    - type: map_at_3
      value: 82.49499999999999
    - type: map_at_5
      value: 84.392
    - type: mrr_at_1
      value: 82.09
    - type: mrr_at_10
      value: 88.301
    - type: mrr_at_100
      value: 88.383
    - type: mrr_at_1000
      value: 88.384
    - type: mrr_at_3
      value: 87.37
    - type: mrr_at_5
      value: 88.035
    - type: ndcg_at_1
      value: 82.12
    - type: ndcg_at_10
      value: 89.149
    - type: ndcg_at_100
      value: 90.235
    - type: ndcg_at_1000
      value: 90.307
    - type: ndcg_at_3
      value: 86.37599999999999
    - type: ndcg_at_5
      value: 87.964
    - type: precision_at_1
      value: 82.12
    - type: precision_at_10
      value: 13.56
    - type: precision_at_100
      value: 1.539
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.88
    - type: precision_at_5
      value: 24.92
    - type: recall_at_1
      value: 71.241
    - type: recall_at_10
      value: 96.128
    - type: recall_at_100
      value: 99.696
    - type: recall_at_1000
      value: 99.994
    - type: recall_at_3
      value: 88.181
    - type: recall_at_5
      value: 92.694
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering
      name: MTEB RedditClustering
      config: default
      split: test
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
    metrics:
    - type: v_measure
      value: 56.59757799655151
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering-p2p
      name: MTEB RedditClusteringP2P
      config: default
      split: test
      revision: 282350215ef01743dc01b456c7f5241fa8937f16
    metrics:
    - type: v_measure
      value: 64.27391998854624
  - task:
      type: Retrieval
    dataset:
      type: scidocs
      name: MTEB SCIDOCS
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 4.243
    - type: map_at_10
      value: 10.965
    - type: map_at_100
      value: 12.934999999999999
    - type: map_at_1000
      value: 13.256
    - type: map_at_3
      value: 7.907
    - type: map_at_5
      value: 9.435
    - type: mrr_at_1
      value: 20.9
    - type: mrr_at_10
      value: 31.849
    - type: mrr_at_100
      value: 32.964
    - type: mrr_at_1000
      value: 33.024
    - type: mrr_at_3
      value: 28.517
    - type: mrr_at_5
      value: 30.381999999999998
    - type: ndcg_at_1
      value: 20.9
    - type: ndcg_at_10
      value: 18.723
    - type: ndcg_at_100
      value: 26.384999999999998
    - type: ndcg_at_1000
      value: 32.114
    - type: ndcg_at_3
      value: 17.753
    - type: ndcg_at_5
      value: 15.558
    - type: precision_at_1
      value: 20.9
    - type: precision_at_10
      value: 9.8
    - type: precision_at_100
      value: 2.078
    - type: precision_at_1000
      value: 0.345
    - type: precision_at_3
      value: 16.900000000000002
    - type: precision_at_5
      value: 13.88
    - type: recall_at_1
      value: 4.243
    - type: recall_at_10
      value: 19.885
    - type: recall_at_100
      value: 42.17
    - type: recall_at_1000
      value: 70.12
    - type: recall_at_3
      value: 10.288
    - type: recall_at_5
      value: 14.072000000000001
  - task:
      type: STS
    dataset:
      type: mteb/sickr-sts
      name: MTEB SICK-R
      config: default
      split: test
      revision: a6ea5a8cab320b040a23452cc28066d9beae2cee
    metrics:
    - type: cos_sim_pearson
      value: 85.84209174935282
    - type: cos_sim_spearman
      value: 81.73248048438833
    - type: euclidean_pearson
      value: 83.02810070308149
    - type: euclidean_spearman
      value: 81.73248295679514
    - type: manhattan_pearson
      value: 82.95368060376002
    - type: manhattan_spearman
      value: 81.60277910998718
  - task:
      type: STS
    dataset:
      type: mteb/sts12-sts
      name: MTEB STS12
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_pearson
      value: 88.52628804556943
    - type: cos_sim_spearman
      value: 82.5713913555672
    - type: euclidean_pearson
      value: 85.8796774746988
    - type: euclidean_spearman
      value: 82.57137506803424
    - type: manhattan_pearson
      value: 85.79671002960058
    - type: manhattan_spearman
      value: 82.49445981618027
  - task:
      type: STS
    dataset:
      type: mteb/sts13-sts
      name: MTEB STS13
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_pearson
      value: 86.23682503505542
    - type: cos_sim_spearman
      value: 87.15008956711806
    - type: euclidean_pearson
      value: 86.79805401524959
    - type: euclidean_spearman
      value: 87.15008956711806
    - type: manhattan_pearson
      value: 86.65298502699244
    - type: manhattan_spearman
      value: 86.97677821948562
  - task:
      type: STS
    dataset:
      type: mteb/sts14-sts
      name: MTEB STS14
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_pearson
      value: 85.63370304677802
    - type: cos_sim_spearman
      value: 84.97105553540318
    - type: euclidean_pearson
      value: 85.28896108687721
    - type: euclidean_spearman
      value: 84.97105553540318
    - type: manhattan_pearson
      value: 85.09663190337331
    - type: manhattan_spearman
      value: 84.79126831644619
  - task:
      type: STS
    dataset:
      type: mteb/sts15-sts
      name: MTEB STS15
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_pearson
      value: 90.2614838800733
    - type: cos_sim_spearman
      value: 91.0509162991835
    - type: euclidean_pearson
      value: 90.33098317533373
    - type: euclidean_spearman
      value: 91.05091625871644
    - type: manhattan_pearson
      value: 90.26250435151107
    - type: manhattan_spearman
      value: 90.97999594417519
  - task:
      type: STS
    dataset:
      type: mteb/sts16-sts
      name: MTEB STS16
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_pearson
      value: 85.80480973335091
    - type: cos_sim_spearman
      value: 87.313695492969
    - type: euclidean_pearson
      value: 86.49267251576939
    - type: euclidean_spearman
      value: 87.313695492969
    - type: manhattan_pearson
      value: 86.44019901831935
    - type: manhattan_spearman
      value: 87.24205395460392
  - task:
      type: STS
    dataset:
      type: mteb/sts17-crosslingual-sts
      name: MTEB STS17 (en-en)
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 90.05662789380672
    - type: cos_sim_spearman
      value: 90.02759424426651
    - type: euclidean_pearson
      value: 90.4042483422981
    - type: euclidean_spearman
      value: 90.02759424426651
    - type: manhattan_pearson
      value: 90.51446975000226
    - type: manhattan_spearman
      value: 90.08832889933616
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (en)
      config: en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 67.5975528273532
    - type: cos_sim_spearman
      value: 67.62969861411354
    - type: euclidean_pearson
      value: 69.224275734323
    - type: euclidean_spearman
      value: 67.62969861411354
    - type: manhattan_pearson
      value: 69.3761447059927
    - type: manhattan_spearman
      value: 67.90921005611467
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_pearson
      value: 87.11244327231684
    - type: cos_sim_spearman
      value: 88.37902438979035
    - type: euclidean_pearson
      value: 87.86054279847336
    - type: euclidean_spearman
      value: 88.37902438979035
    - type: manhattan_pearson
      value: 87.77257757320378
    - type: manhattan_spearman
      value: 88.25208966098123
  - task:
      type: Reranking
    dataset:
      type: mteb/scidocs-reranking
      name: MTEB SciDocsRR
      config: default
      split: test
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
    metrics:
    - type: map
      value: 85.87174608143563
    - type: mrr
      value: 96.12836872640794
  - task:
      type: Retrieval
    dataset:
      type: scifact
      name: MTEB SciFact
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 57.760999999999996
    - type: map_at_10
      value: 67.258
    - type: map_at_100
      value: 67.757
    - type: map_at_1000
      value: 67.78800000000001
    - type: map_at_3
      value: 64.602
    - type: map_at_5
      value: 65.64
    - type: mrr_at_1
      value: 60.667
    - type: mrr_at_10
      value: 68.441
    - type: mrr_at_100
      value: 68.825
    - type: mrr_at_1000
      value: 68.853
    - type: mrr_at_3
      value: 66.444
    - type: mrr_at_5
      value: 67.26100000000001
    - type: ndcg_at_1
      value: 60.667
    - type: ndcg_at_10
      value: 71.852
    - type: ndcg_at_100
      value: 73.9
    - type: ndcg_at_1000
      value: 74.628
    - type: ndcg_at_3
      value: 67.093
    - type: ndcg_at_5
      value: 68.58
    - type: precision_at_1
      value: 60.667
    - type: precision_at_10
      value: 9.6
    - type: precision_at_100
      value: 1.0670000000000002
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 26.111
    - type: precision_at_5
      value: 16.733
    - type: recall_at_1
      value: 57.760999999999996
    - type: recall_at_10
      value: 84.967
    - type: recall_at_100
      value: 93.833
    - type: recall_at_1000
      value: 99.333
    - type: recall_at_3
      value: 71.589
    - type: recall_at_5
      value: 75.483
  - task:
      type: PairClassification
    dataset:
      type: mteb/sprintduplicatequestions-pairclassification
      name: MTEB SprintDuplicateQuestions
      config: default
      split: test
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
    metrics:
    - type: cos_sim_accuracy
      value: 99.66633663366336
    - type: cos_sim_ap
      value: 91.17685358899108
    - type: cos_sim_f1
      value: 82.16818642350559
    - type: cos_sim_precision
      value: 83.26488706365504
    - type: cos_sim_recall
      value: 81.10000000000001
    - type: dot_accuracy
      value: 99.66633663366336
    - type: dot_ap
      value: 91.17663411119032
    - type: dot_f1
      value: 82.16818642350559
    - type: dot_precision
      value: 83.26488706365504
    - type: dot_recall
      value: 81.10000000000001
    - type: euclidean_accuracy
      value: 99.66633663366336
    - type: euclidean_ap
      value: 91.17685189882275
    - type: euclidean_f1
      value: 82.16818642350559
    - type: euclidean_precision
      value: 83.26488706365504
    - type: euclidean_recall
      value: 81.10000000000001
    - type: manhattan_accuracy
      value: 99.66633663366336
    - type: manhattan_ap
      value: 91.2241619496737
    - type: manhattan_f1
      value: 82.20472440944883
    - type: manhattan_precision
      value: 86.51933701657458
    - type: manhattan_recall
      value: 78.3
    - type: max_accuracy
      value: 99.66633663366336
    - type: max_ap
      value: 91.2241619496737
    - type: max_f1
      value: 82.20472440944883
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering
      name: MTEB StackExchangeClustering
      config: default
      split: test
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
    metrics:
    - type: v_measure
      value: 66.85101268897951
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering-p2p
      name: MTEB StackExchangeClusteringP2P
      config: default
      split: test
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
    metrics:
    - type: v_measure
      value: 42.461184054706905
  - task:
      type: Reranking
    dataset:
      type: mteb/stackoverflowdupquestions-reranking
      name: MTEB StackOverflowDupQuestions
      config: default
      split: test
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
    metrics:
    - type: map
      value: 51.44542568873886
    - type: mrr
      value: 52.33656151854681
  - task:
      type: Summarization
    dataset:
      type: mteb/summeval
      name: MTEB SummEval
      config: default
      split: test
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
    metrics:
    - type: cos_sim_pearson
      value: 30.75982974997539
    - type: cos_sim_spearman
      value: 30.385405026539914
    - type: dot_pearson
      value: 30.75982433546523
    - type: dot_spearman
      value: 30.385405026539914
  - task:
      type: Retrieval
    dataset:
      type: trec-covid
      name: MTEB TRECCOVID
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 0.22799999999999998
    - type: map_at_10
      value: 2.064
    - type: map_at_100
      value: 13.056000000000001
    - type: map_at_1000
      value: 31.747999999999998
    - type: map_at_3
      value: 0.67
    - type: map_at_5
      value: 1.097
    - type: mrr_at_1
      value: 90.0
    - type: mrr_at_10
      value: 94.667
    - type: mrr_at_100
      value: 94.667
    - type: mrr_at_1000
      value: 94.667
    - type: mrr_at_3
      value: 94.667
    - type: mrr_at_5
      value: 94.667
    - type: ndcg_at_1
      value: 86.0
    - type: ndcg_at_10
      value: 82.0
    - type: ndcg_at_100
      value: 64.307
    - type: ndcg_at_1000
      value: 57.023999999999994
    - type: ndcg_at_3
      value: 85.816
    - type: ndcg_at_5
      value: 84.904
    - type: precision_at_1
      value: 90.0
    - type: precision_at_10
      value: 85.8
    - type: precision_at_100
      value: 66.46
    - type: precision_at_1000
      value: 25.202
    - type: precision_at_3
      value: 90.0
    - type: precision_at_5
      value: 89.2
    - type: recall_at_1
      value: 0.22799999999999998
    - type: recall_at_10
      value: 2.235
    - type: recall_at_100
      value: 16.185
    - type: recall_at_1000
      value: 53.620999999999995
    - type: recall_at_3
      value: 0.7040000000000001
    - type: recall_at_5
      value: 1.172
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (sqi-eng)
      config: sqi-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.39999999999999
    - type: f1
      value: 96.75
    - type: precision
      value: 96.45
    - type: recall
      value: 97.39999999999999
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (fry-eng)
      config: fry-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 85.54913294797689
    - type: f1
      value: 82.46628131021194
    - type: precision
      value: 81.1175337186898
    - type: recall
      value: 85.54913294797689
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (kur-eng)
      config: kur-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 81.21951219512195
    - type: f1
      value: 77.33333333333334
    - type: precision
      value: 75.54878048780488
    - type: recall
      value: 81.21951219512195
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (tur-eng)
      config: tur-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 98.6
    - type: f1
      value: 98.26666666666665
    - type: precision
      value: 98.1
    - type: recall
      value: 98.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (deu-eng)
      config: deu-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 99.5
    - type: f1
      value: 99.33333333333333
    - type: precision
      value: 99.25
    - type: recall
      value: 99.5
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (nld-eng)
      config: nld-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.8
    - type: f1
      value: 97.2
    - type: precision
      value: 96.89999999999999
    - type: recall
      value: 97.8
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ron-eng)
      config: ron-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.8
    - type: f1
      value: 97.18333333333334
    - type: precision
      value: 96.88333333333333
    - type: recall
      value: 97.8
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ang-eng)
      config: ang-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 77.61194029850746
    - type: f1
      value: 72.81094527363183
    - type: precision
      value: 70.83333333333333
    - type: recall
      value: 77.61194029850746
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ido-eng)
      config: ido-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.7
    - type: f1
      value: 91.91666666666667
    - type: precision
      value: 91.08333333333334
    - type: recall
      value: 93.7
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (jav-eng)
      config: jav-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 88.29268292682927
    - type: f1
      value: 85.27642276422765
    - type: precision
      value: 84.01277584204414
    - type: recall
      value: 88.29268292682927
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (isl-eng)
      config: isl-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.1
    - type: f1
      value: 95.0
    - type: precision
      value: 94.46666666666668
    - type: recall
      value: 96.1
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (slv-eng)
      config: slv-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.681652490887
    - type: f1
      value: 91.90765492102065
    - type: precision
      value: 91.05913325232888
    - type: recall
      value: 93.681652490887
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (cym-eng)
      config: cym-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.17391304347827
    - type: f1
      value: 89.97101449275361
    - type: precision
      value: 88.96811594202899
    - type: recall
      value: 92.17391304347827
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (kaz-eng)
      config: kaz-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.43478260869566
    - type: f1
      value: 87.72173913043478
    - type: precision
      value: 86.42028985507245
    - type: recall
      value: 90.43478260869566
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (est-eng)
      config: est-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.4
    - type: f1
      value: 88.03
    - type: precision
      value: 86.95
    - type: recall
      value: 90.4
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (heb-eng)
      config: heb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.4
    - type: f1
      value: 91.45666666666666
    - type: precision
      value: 90.525
    - type: recall
      value: 93.4
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (gla-eng)
      config: gla-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 81.9059107358263
    - type: f1
      value: 78.32557872364869
    - type: precision
      value: 76.78260286824823
    - type: recall
      value: 81.9059107358263
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (mar-eng)
      config: mar-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.3
    - type: f1
      value: 92.58333333333333
    - type: precision
      value: 91.73333333333332
    - type: recall
      value: 94.3
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (lat-eng)
      config: lat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 79.10000000000001
    - type: f1
      value: 74.50500000000001
    - type: precision
      value: 72.58928571428571
    - type: recall
      value: 79.10000000000001
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (bel-eng)
      config: bel-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.6
    - type: f1
      value: 95.55
    - type: precision
      value: 95.05
    - type: recall
      value: 96.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (pms-eng)
      config: pms-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 82.0952380952381
    - type: f1
      value: 77.98458049886621
    - type: precision
      value: 76.1968253968254
    - type: recall
      value: 82.0952380952381
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (gle-eng)
      config: gle-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87.9
    - type: f1
      value: 84.99190476190476
    - type: precision
      value: 83.65
    - type: recall
      value: 87.9
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (pes-eng)
      config: pes-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.7
    - type: f1
      value: 94.56666666666666
    - type: precision
      value: 94.01666666666667
    - type: recall
      value: 95.7
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (nob-eng)
      config: nob-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 98.6
    - type: f1
      value: 98.2
    - type: precision
      value: 98.0
    - type: recall
      value: 98.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (bul-eng)
      config: bul-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.6
    - type: f1
      value: 94.38333333333334
    - type: precision
      value: 93.78333333333335
    - type: recall
      value: 95.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (cbk-eng)
      config: cbk-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87.4
    - type: f1
      value: 84.10380952380952
    - type: precision
      value: 82.67
    - type: recall
      value: 87.4
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (hun-eng)
      config: hun-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.5
    - type: f1
      value: 94.33333333333334
    - type: precision
      value: 93.78333333333333
    - type: recall
      value: 95.5
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (uig-eng)
      config: uig-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.4
    - type: f1
      value: 86.82000000000001
    - type: precision
      value: 85.64500000000001
    - type: recall
      value: 89.4
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (rus-eng)
      config: rus-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.1
    - type: f1
      value: 93.56666666666668
    - type: precision
      value: 92.81666666666666
    - type: recall
      value: 95.1
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (spa-eng)
      config: spa-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 98.9
    - type: f1
      value: 98.6
    - type: precision
      value: 98.45
    - type: recall
      value: 98.9
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (hye-eng)
      config: hye-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.01347708894879
    - type: f1
      value: 93.51752021563343
    - type: precision
      value: 92.82794249775381
    - type: recall
      value: 95.01347708894879
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (tel-eng)
      config: tel-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.00854700854701
    - type: f1
      value: 96.08262108262107
    - type: precision
      value: 95.65527065527067
    - type: recall
      value: 97.00854700854701
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (afr-eng)
      config: afr-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.5
    - type: f1
      value: 95.39999999999999
    - type: precision
      value: 94.88333333333333
    - type: recall
      value: 96.5
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (mon-eng)
      config: mon-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.5909090909091
    - type: f1
      value: 95.49242424242425
    - type: precision
      value: 94.9621212121212
    - type: recall
      value: 96.5909090909091
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (arz-eng)
      config: arz-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 84.90566037735849
    - type: f1
      value: 81.85883997204752
    - type: precision
      value: 80.54507337526205
    - type: recall
      value: 84.90566037735849
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (hrv-eng)
      config: hrv-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.5
    - type: f1
      value: 96.75
    - type: precision
      value: 96.38333333333333
    - type: recall
      value: 97.5
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (nov-eng)
      config: nov-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 86.7704280155642
    - type: f1
      value: 82.99610894941635
    - type: precision
      value: 81.32295719844358
    - type: recall
      value: 86.7704280155642
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (gsw-eng)
      config: gsw-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 67.52136752136752
    - type: f1
      value: 61.89662189662191
    - type: precision
      value: 59.68660968660969
    - type: recall
      value: 67.52136752136752
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (nds-eng)
      config: nds-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.2
    - type: f1
      value: 86.32
    - type: precision
      value: 85.015
    - type: recall
      value: 89.2
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ukr-eng)
      config: ukr-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.0
    - type: f1
      value: 94.78333333333333
    - type: precision
      value: 94.18333333333334
    - type: recall
      value: 96.0
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (uzb-eng)
      config: uzb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 83.8785046728972
    - type: f1
      value: 80.54517133956385
    - type: precision
      value: 79.154984423676
    - type: recall
      value: 83.8785046728972
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (lit-eng)
      config: lit-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.60000000000001
    - type: f1
      value: 92.01333333333334
    - type: precision
      value: 91.28333333333333
    - type: recall
      value: 93.60000000000001
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ina-eng)
      config: ina-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.1
    - type: f1
      value: 96.26666666666667
    - type: precision
      value: 95.85000000000001
    - type: recall
      value: 97.1
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (lfn-eng)
      config: lfn-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 84.3
    - type: f1
      value: 80.67833333333333
    - type: precision
      value: 79.03928571428571
    - type: recall
      value: 84.3
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (zsm-eng)
      config: zsm-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.3
    - type: f1
      value: 96.48333333333332
    - type: precision
      value: 96.08333333333331
    - type: recall
      value: 97.3
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ita-eng)
      config: ita-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.7
    - type: f1
      value: 94.66666666666667
    - type: precision
      value: 94.16666666666667
    - type: recall
      value: 95.7
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (cmn-eng)
      config: cmn-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.2
    - type: f1
      value: 96.36666666666667
    - type: precision
      value: 95.96666666666668
    - type: recall
      value: 97.2
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (lvs-eng)
      config: lvs-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.3
    - type: f1
      value: 92.80666666666667
    - type: precision
      value: 92.12833333333333
    - type: recall
      value: 94.3
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (glg-eng)
      config: glg-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.0
    - type: f1
      value: 96.22333333333334
    - type: precision
      value: 95.875
    - type: recall
      value: 97.0
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ceb-eng)
      config: ceb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 74.33333333333333
    - type: f1
      value: 70.78174603174602
    - type: precision
      value: 69.28333333333332
    - type: recall
      value: 74.33333333333333
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (bre-eng)
      config: bre-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 37.6
    - type: f1
      value: 32.938348952090365
    - type: precision
      value: 31.2811038961039
    - type: recall
      value: 37.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ben-eng)
      config: ben-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 91.5
    - type: f1
      value: 89.13333333333333
    - type: precision
      value: 88.03333333333333
    - type: recall
      value: 91.5
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (swg-eng)
      config: swg-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 82.14285714285714
    - type: f1
      value: 77.67857142857143
    - type: precision
      value: 75.59523809523809
    - type: recall
      value: 82.14285714285714
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (arq-eng)
      config: arq-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 69.0450054884742
    - type: f1
      value: 63.070409283362075
    - type: precision
      value: 60.58992781824835
    - type: recall
      value: 69.0450054884742
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (kab-eng)
      config: kab-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 63.1
    - type: f1
      value: 57.848333333333336
    - type: precision
      value: 55.69500000000001
    - type: recall
      value: 63.1
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (fra-eng)
      config: fra-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.1
    - type: f1
      value: 95.01666666666667
    - type: precision
      value: 94.5
    - type: recall
      value: 96.1
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (por-eng)
      config: por-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.89999999999999
    - type: f1
      value: 94.90666666666667
    - type: precision
      value: 94.425
    - type: recall
      value: 95.89999999999999
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (tat-eng)
      config: tat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87.6
    - type: f1
      value: 84.61333333333333
    - type: precision
      value: 83.27
    - type: recall
      value: 87.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (oci-eng)
      config: oci-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 76.4
    - type: f1
      value: 71.90746031746032
    - type: precision
      value: 70.07027777777778
    - type: recall
      value: 76.4
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (pol-eng)
      config: pol-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.89999999999999
    - type: f1
      value: 97.26666666666667
    - type: precision
      value: 96.95
    - type: recall
      value: 97.89999999999999
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (war-eng)
      config: war-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 78.8
    - type: f1
      value: 74.39555555555555
    - type: precision
      value: 72.59416666666667
    - type: recall
      value: 78.8
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (aze-eng)
      config: aze-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.19999999999999
    - type: f1
      value: 93.78999999999999
    - type: precision
      value: 93.125
    - type: recall
      value: 95.19999999999999
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (vie-eng)
      config: vie-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.8
    - type: f1
      value: 97.1
    - type: precision
      value: 96.75
    - type: recall
      value: 97.8
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (nno-eng)
      config: nno-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.6
    - type: f1
      value: 94.25666666666666
    - type: precision
      value: 93.64166666666668
    - type: recall
      value: 95.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (cha-eng)
      config: cha-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 56.934306569343065
    - type: f1
      value: 51.461591936044485
    - type: precision
      value: 49.37434827945776
    - type: recall
      value: 56.934306569343065
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (mhr-eng)
      config: mhr-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 20.200000000000003
    - type: f1
      value: 16.91799284049284
    - type: precision
      value: 15.791855158730158
    - type: recall
      value: 20.200000000000003
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (dan-eng)
      config: dan-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.2
    - type: f1
      value: 95.3
    - type: precision
      value: 94.85
    - type: recall
      value: 96.2
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ell-eng)
      config: ell-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.3
    - type: f1
      value: 95.11666666666667
    - type: precision
      value: 94.53333333333333
    - type: recall
      value: 96.3
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (amh-eng)
      config: amh-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.88095238095238
    - type: f1
      value: 87.14285714285714
    - type: precision
      value: 85.96230158730161
    - type: recall
      value: 89.88095238095238
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (pam-eng)
      config: pam-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 24.099999999999998
    - type: f1
      value: 19.630969083349783
    - type: precision
      value: 18.275094905094907
    - type: recall
      value: 24.099999999999998
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (hsb-eng)
      config: hsb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 83.4368530020704
    - type: f1
      value: 79.45183870649709
    - type: precision
      value: 77.7432712215321
    - type: recall
      value: 83.4368530020704
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (srp-eng)
      config: srp-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.8
    - type: f1
      value: 94.53333333333333
    - type: precision
      value: 93.91666666666666
    - type: recall
      value: 95.8
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (epo-eng)
      config: epo-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 98.8
    - type: f1
      value: 98.48333333333332
    - type: precision
      value: 98.33333333333334
    - type: recall
      value: 98.8
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (kzj-eng)
      config: kzj-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 17.5
    - type: f1
      value: 14.979285714285714
    - type: precision
      value: 14.23235060690943
    - type: recall
      value: 17.5
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (awa-eng)
      config: awa-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.93939393939394
    - type: f1
      value: 91.991341991342
    - type: precision
      value: 91.05339105339105
    - type: recall
      value: 93.93939393939394
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (fao-eng)
      config: fao-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 89.31297709923665
    - type: f1
      value: 86.76844783715012
    - type: precision
      value: 85.63613231552164
    - type: recall
      value: 89.31297709923665
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (mal-eng)
      config: mal-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 99.12663755458514
    - type: f1
      value: 98.93255701115964
    - type: precision
      value: 98.83551673944687
    - type: recall
      value: 99.12663755458514
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ile-eng)
      config: ile-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.0
    - type: f1
      value: 89.77999999999999
    - type: precision
      value: 88.78333333333333
    - type: recall
      value: 92.0
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (bos-eng)
      config: bos-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.89265536723164
    - type: f1
      value: 95.85687382297553
    - type: precision
      value: 95.33898305084746
    - type: recall
      value: 96.89265536723164
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (cor-eng)
      config: cor-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 14.6
    - type: f1
      value: 11.820611790170615
    - type: precision
      value: 11.022616224355355
    - type: recall
      value: 14.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (cat-eng)
      config: cat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.89999999999999
    - type: f1
      value: 94.93333333333334
    - type: precision
      value: 94.48666666666666
    - type: recall
      value: 95.89999999999999
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (eus-eng)
      config: eus-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 87.6
    - type: f1
      value: 84.72333333333334
    - type: precision
      value: 83.44166666666666
    - type: recall
      value: 87.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (yue-eng)
      config: yue-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.8
    - type: f1
      value: 93.47333333333333
    - type: precision
      value: 92.875
    - type: recall
      value: 94.8
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (swe-eng)
      config: swe-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.6
    - type: f1
      value: 95.71666666666665
    - type: precision
      value: 95.28333333333335
    - type: recall
      value: 96.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (dtp-eng)
      config: dtp-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 17.8
    - type: f1
      value: 14.511074040901628
    - type: precision
      value: 13.503791000666002
    - type: recall
      value: 17.8
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (kat-eng)
      config: kat-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.10187667560321
    - type: f1
      value: 92.46648793565683
    - type: precision
      value: 91.71134941912423
    - type: recall
      value: 94.10187667560321
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (jpn-eng)
      config: jpn-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.0
    - type: f1
      value: 96.11666666666666
    - type: precision
      value: 95.68333333333334
    - type: recall
      value: 97.0
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (csb-eng)
      config: csb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 72.72727272727273
    - type: f1
      value: 66.58949745906267
    - type: precision
      value: 63.86693017127799
    - type: recall
      value: 72.72727272727273
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (xho-eng)
      config: xho-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 90.14084507042254
    - type: f1
      value: 88.26291079812206
    - type: precision
      value: 87.32394366197182
    - type: recall
      value: 90.14084507042254
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (orv-eng)
      config: orv-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 64.67065868263472
    - type: f1
      value: 58.2876627696987
    - type: precision
      value: 55.79255774165953
    - type: recall
      value: 64.67065868263472
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ind-eng)
      config: ind-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 95.6
    - type: f1
      value: 94.41666666666667
    - type: precision
      value: 93.85
    - type: recall
      value: 95.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (tuk-eng)
      config: tuk-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 55.172413793103445
    - type: f1
      value: 49.63992493549144
    - type: precision
      value: 47.71405113769646
    - type: recall
      value: 55.172413793103445
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (max-eng)
      config: max-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 77.46478873239437
    - type: f1
      value: 73.4417616811983
    - type: precision
      value: 71.91607981220658
    - type: recall
      value: 77.46478873239437
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (swh-eng)
      config: swh-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 84.61538461538461
    - type: f1
      value: 80.91452991452994
    - type: precision
      value: 79.33760683760683
    - type: recall
      value: 84.61538461538461
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (hin-eng)
      config: hin-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 98.2
    - type: f1
      value: 97.6
    - type: precision
      value: 97.3
    - type: recall
      value: 98.2
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (dsb-eng)
      config: dsb-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 75.5741127348643
    - type: f1
      value: 72.00417536534445
    - type: precision
      value: 70.53467872883321
    - type: recall
      value: 75.5741127348643
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ber-eng)
      config: ber-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 62.2
    - type: f1
      value: 55.577460317460314
    - type: precision
      value: 52.98583333333333
    - type: recall
      value: 62.2
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (tam-eng)
      config: tam-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.18241042345277
    - type: f1
      value: 90.6468124709167
    - type: precision
      value: 89.95656894679696
    - type: recall
      value: 92.18241042345277
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (slk-eng)
      config: slk-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.1
    - type: f1
      value: 95.13333333333333
    - type: precision
      value: 94.66666666666667
    - type: recall
      value: 96.1
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (tgl-eng)
      config: tgl-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 96.8
    - type: f1
      value: 95.85000000000001
    - type: precision
      value: 95.39999999999999
    - type: recall
      value: 96.8
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ast-eng)
      config: ast-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.1259842519685
    - type: f1
      value: 89.76377952755905
    - type: precision
      value: 88.71391076115485
    - type: recall
      value: 92.1259842519685
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (mkd-eng)
      config: mkd-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.1
    - type: f1
      value: 92.49
    - type: precision
      value: 91.725
    - type: recall
      value: 94.1
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (khm-eng)
      config: khm-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 77.5623268698061
    - type: f1
      value: 73.27364463791058
    - type: precision
      value: 71.51947852086357
    - type: recall
      value: 77.5623268698061
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ces-eng)
      config: ces-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.39999999999999
    - type: f1
      value: 96.56666666666666
    - type: precision
      value: 96.16666666666667
    - type: recall
      value: 97.39999999999999
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (tzl-eng)
      config: tzl-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 66.34615384615384
    - type: f1
      value: 61.092032967032964
    - type: precision
      value: 59.27197802197802
    - type: recall
      value: 66.34615384615384
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (urd-eng)
      config: urd-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.89999999999999
    - type: f1
      value: 93.41190476190476
    - type: precision
      value: 92.7
    - type: recall
      value: 94.89999999999999
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (ara-eng)
      config: ara-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.10000000000001
    - type: f1
      value: 91.10000000000001
    - type: precision
      value: 90.13333333333333
    - type: recall
      value: 93.10000000000001
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (kor-eng)
      config: kor-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 93.7
    - type: f1
      value: 91.97333333333334
    - type: precision
      value: 91.14166666666667
    - type: recall
      value: 93.7
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (yid-eng)
      config: yid-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 92.21698113207547
    - type: f1
      value: 90.3796046720575
    - type: precision
      value: 89.56367924528303
    - type: recall
      value: 92.21698113207547
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (fin-eng)
      config: fin-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.6
    - type: f1
      value: 96.91666666666667
    - type: precision
      value: 96.6
    - type: recall
      value: 97.6
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (tha-eng)
      config: tha-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 97.44525547445255
    - type: f1
      value: 96.71532846715328
    - type: precision
      value: 96.35036496350365
    - type: recall
      value: 97.44525547445255
  - task:
      type: BitextMining
    dataset:
      type: mteb/tatoeba-bitext-mining
      name: MTEB Tatoeba (wuu-eng)
      config: wuu-eng
      split: test
      revision: 9080400076fbadbb4c4dcb136ff4eddc40b42553
    metrics:
    - type: accuracy
      value: 94.1
    - type: f1
      value: 92.34000000000002
    - type: precision
      value: 91.49166666666667
    - type: recall
      value: 94.1
  - task:
      type: Retrieval
    dataset:
      type: webis-touche2020
      name: MTEB Touche2020
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 3.2910000000000004
    - type: map_at_10
      value: 10.373000000000001
    - type: map_at_100
      value: 15.612
    - type: map_at_1000
      value: 17.06
    - type: map_at_3
      value: 6.119
    - type: map_at_5
      value: 7.917000000000001
    - type: mrr_at_1
      value: 44.897999999999996
    - type: mrr_at_10
      value: 56.054
    - type: mrr_at_100
      value: 56.82000000000001
    - type: mrr_at_1000
      value: 56.82000000000001
    - type: mrr_at_3
      value: 52.381
    - type: mrr_at_5
      value: 53.81
    - type: ndcg_at_1
      value: 42.857
    - type: ndcg_at_10
      value: 27.249000000000002
    - type: ndcg_at_100
      value: 36.529
    - type: ndcg_at_1000
      value: 48.136
    - type: ndcg_at_3
      value: 33.938
    - type: ndcg_at_5
      value: 29.951
    - type: precision_at_1
      value: 44.897999999999996
    - type: precision_at_10
      value: 22.653000000000002
    - type: precision_at_100
      value: 7.000000000000001
    - type: precision_at_1000
      value: 1.48
    - type: precision_at_3
      value: 32.653
    - type: precision_at_5
      value: 27.755000000000003
    - type: recall_at_1
      value: 3.2910000000000004
    - type: recall_at_10
      value: 16.16
    - type: recall_at_100
      value: 43.908
    - type: recall_at_1000
      value: 79.823
    - type: recall_at_3
      value: 7.156
    - type: recall_at_5
      value: 10.204
  - task:
      type: Classification
    dataset:
      type: mteb/toxic_conversations_50k
      name: MTEB ToxicConversationsClassification
      config: default
      split: test
      revision: d7c0de2777da35d6aae2200a62c6e0e5af397c4c
    metrics:
    - type: accuracy
      value: 71.05879999999999
    - type: ap
      value: 14.609748142799111
    - type: f1
      value: 54.878956295843096
  - task:
      type: Classification
    dataset:
      type: mteb/tweet_sentiment_extraction
      name: MTEB TweetSentimentExtractionClassification
      config: default
      split: test
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
    metrics:
    - type: accuracy
      value: 64.61799660441426
    - type: f1
      value: 64.8698191961434
  - task:
      type: Clustering
    dataset:
      type: mteb/twentynewsgroups-clustering
      name: MTEB TwentyNewsgroupsClustering
      config: default
      split: test
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
    metrics:
    - type: v_measure
      value: 51.32860036611885
  - task:
      type: PairClassification
    dataset:
      type: mteb/twittersemeval2015-pairclassification
      name: MTEB TwitterSemEval2015
      config: default
      split: test
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
    metrics:
    - type: cos_sim_accuracy
      value: 88.34714192048638
    - type: cos_sim_ap
      value: 80.26732975975634
    - type: cos_sim_f1
      value: 73.53415148134374
    - type: cos_sim_precision
      value: 69.34767360299276
    - type: cos_sim_recall
      value: 78.25857519788919
    - type: dot_accuracy
      value: 88.34714192048638
    - type: dot_ap
      value: 80.26733698491206
    - type: dot_f1
      value: 73.53415148134374
    - type: dot_precision
      value: 69.34767360299276
    - type: dot_recall
      value: 78.25857519788919
    - type: euclidean_accuracy
      value: 88.34714192048638
    - type: euclidean_ap
      value: 80.26734337771738
    - type: euclidean_f1
      value: 73.53415148134374
    - type: euclidean_precision
      value: 69.34767360299276
    - type: euclidean_recall
      value: 78.25857519788919
    - type: manhattan_accuracy
      value: 88.30541813196639
    - type: manhattan_ap
      value: 80.19415808104145
    - type: manhattan_f1
      value: 73.55143870713441
    - type: manhattan_precision
      value: 73.25307511122743
    - type: manhattan_recall
      value: 73.85224274406332
    - type: max_accuracy
      value: 88.34714192048638
    - type: max_ap
      value: 80.26734337771738
    - type: max_f1
      value: 73.55143870713441
  - task:
      type: PairClassification
    dataset:
      type: mteb/twitterurlcorpus-pairclassification
      name: MTEB TwitterURLCorpus
      config: default
      split: test
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
    metrics:
    - type: cos_sim_accuracy
      value: 89.81061047075717
    - type: cos_sim_ap
      value: 87.11747055081017
    - type: cos_sim_f1
      value: 80.04355498817256
    - type: cos_sim_precision
      value: 78.1165262000733
    - type: cos_sim_recall
      value: 82.06806282722513
    - type: dot_accuracy
      value: 89.81061047075717
    - type: dot_ap
      value: 87.11746902745236
    - type: dot_f1
      value: 80.04355498817256
    - type: dot_precision
      value: 78.1165262000733
    - type: dot_recall
      value: 82.06806282722513
    - type: euclidean_accuracy
      value: 89.81061047075717
    - type: euclidean_ap
      value: 87.11746919324248
    - type: euclidean_f1
      value: 80.04355498817256
    - type: euclidean_precision
      value: 78.1165262000733
    - type: euclidean_recall
      value: 82.06806282722513
    - type: manhattan_accuracy
      value: 89.79508673885202
    - type: manhattan_ap
      value: 87.11074390832218
    - type: manhattan_f1
      value: 80.13002540726349
    - type: manhattan_precision
      value: 77.83826945412311
    - type: manhattan_recall
      value: 82.56082537727133
    - type: max_accuracy
      value: 89.81061047075717
    - type: max_ap
      value: 87.11747055081017
    - type: max_f1
      value: 80.13002540726349
language:
- multilingual
- af
- am
- ar
- as
- az
- be
- bg
- bn
- br
- bs
- ca
- cs
- cy
- da
- de
- el
- en
- eo
- es
- et
- eu
- fa
- fi
- fr
- fy
- ga
- gd
- gl
- gu
- ha
- he
- hi
- hr
- hu
- hy
- id
- is
- it
- ja
- jv
- ka
- kk
- km
- kn
- ko
- ku
- ky
- la
- lo
- lt
- lv
- mg
- mk
- ml
- mn
- mr
- ms
- my
- ne
- nl
- 'no'
- om
- or
- pa
- pl
- ps
- pt
- ro
- ru
- sa
- sd
- si
- sk
- sl
- so
- sq
- sr
- su
- sv
- sw
- ta
- te
- th
- tl
- tr
- ug
- uk
- ur
- uz
- vi
- xh
- yi
- zh
license: mit
---

## Multilingual-E5-large-instruct

[Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/pdf/2402.05672).
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei, arXiv 2024

This model has 24 layers and the embedding size is 1024.

## Usage

Below are examples to encode queries and passages from the MS-MARCO passage ranking dataset.

### Transformers

```python
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, '南瓜的家常做法')
]
# No need to add instruction for retrieval documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
]
input_texts = queries + documents

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
# => [[91.92852783203125, 67.580322265625], [70.3814468383789, 92.1330795288086]]
```

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, '南瓜的家常做法')
]
# No need to add instruction for retrieval documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
]
input_texts = queries + documents

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

embeddings = model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
# [[91.92853546142578, 67.5802993774414], [70.38143157958984, 92.13307189941406]]
```

### Infinity

Usage with [Infinity](https://github.com/michaelfeil/infinity):

```bash
docker run --gpus all -v $PWD/data:/app/.cache -e HF_TOKEN=$HF_TOKEN -p "7997":"7997" \
michaelf34/infinity:0.0.68 \
v2 --model-id intfloat/multilingual-e5-large-instruct --revision "main" --dtype float16 --batch-size 32 -engine torch --port 7997
```

## Supported Languages

This model is initialized from [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)
and continually trained on a mixture of multilingual datasets.
It supports 100 languages from xlm-roberta,
but low-resource languages may see performance degradation.

## Training Details

**Initialization**: [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)

**First stage**: contrastive pre-training with 1 billion weakly supervised text pairs.

**Second stage**: fine-tuning on datasets from the [E5-mistral](https://arxiv.org/abs/2401.00368) paper.

## MTEB Benchmark Evaluation

Check out [unilm/e5](https://github.com/microsoft/unilm/tree/master/e5) to reproduce evaluation results 
on the [BEIR](https://arxiv.org/abs/2104.08663) and [MTEB benchmark](https://arxiv.org/abs/2210.07316).

## FAQ

**1. Do I need to add instructions to the query?**

Yes, this is how the model is trained, otherwise you will see a performance degradation.
The task definition should be a one-sentence instruction that describes the task.
This is a way to customize text embeddings for different scenarios through natural language instructions.

Please check out [unilm/e5/utils.py](https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106) for instructions we used for evaluation.

On the other hand, there is no need to add instructions to the document side.

**2. Why are my reproduced results slightly different from reported in the model card?**

Different versions of `transformers` and `pytorch` could cause negligible but non-zero performance differences.

**3. Why does the cosine similarity scores distribute around 0.7 to 1.0?**

This is a known and expected behavior as we use a low temperature 0.01 for InfoNCE contrastive loss. 

For text embedding tasks like text retrieval or semantic similarity, 
what matters is the relative order of the scores instead of the absolute values, 
so this should not be an issue.

## Citation

If you find our paper or models helpful, please consider cite as follows:

```
@article{wang2024multilingual,
  title={Multilingual E5 Text Embeddings: A Technical Report},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2402.05672},
  year={2024}
}
```

## Limitations

Long texts will be truncated to at most 512 tokens.
