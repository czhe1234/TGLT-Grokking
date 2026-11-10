# 文件来源

source_files.csv记录每个原文件的服务器相对路径、原始字节数与SHA-256，以及包内路径与SHA-256。
原始服务器根目录：/home/chenzhe/work/tglt_experiments。
history.json.gz为无损gzip封装；解压后字节与原history.json一致。
continuation_config.json保留续训设置，history_100k_original.json.gz保留未续训前原记录。
experiments/run_index.csv是本包可直接使用的相对路径入口。

historical_scripts保存产生已有结果的原脚本。它们包含历史目录和环境依赖，用于核对公式和追溯实现，不承诺脱离原服务器直接执行。
原稿sn-article.tex和大型检查点本次未打包。原稿与第一代实现存在差异，避免把草稿实验设定直接当成这些结果的实施说明。
新写的scripts只核对包内文件、执行CPU模型检查、重画保存数值；不启动训练。

未附开源许可证。仓库保持私人。
