# Configuration file for the Sphinx documentation builder.
# Sphinx用のドキュメントビルダーの設定ファイル。

#
# For the full list of built-in configuration values, see the documentation:
# 組み込みの設定値の完全なリストは、以下のドキュメントを参照してください。
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# プロジェクトに関する情報のセクション。

# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# プロジェクト情報の設定に関するドキュメント。

project = 'nuplan-devkit'  # プロジェクト名を設定。
copyright = '2022, patk'  # 著作権情報を設定。
author = 'patk'  # 著者名を設定。
release = 'v0.1'  # リリースバージョンを設定。

# -- General configuration ---------------------------------------------------
# 一般的な設定に関するセクション。

# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# 一般設定に関するドキュメントへのリンク。

extensions = [
    "sphinx.ext.duration",  # ドキュメントのビルド時間を測定する拡張。
    "sphinx.ext.doctest",  # ドキュメント内のdoctestコードブロックをテストする拡張。
    "sphinx.ext.autodoc",  # Pythonモジュールの自動ドキュメント生成を提供する拡張。
    "sphinx.ext.autosummary",  # サマリーテーブルを自動生成する拡張。
    "sphinx.ext.intersphinx",  # 他のSphinxプロジェクトのドキュメントを参照する拡張。
    "myst_parser",  # Markdownのサポートを追加する拡張。
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),  # Read the Docsのドキュメントを参照。
    "python": ("https://docs.python.org/3/", None),  # Pythonの公式ドキュメントを参照。
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),  # Sphinxの公式ドキュメントを参照。
}
intersphinx_disabled_domains = ["std"]  # 標準ドメインを無効化。

templates_path = ["_templates"]  # カスタムテンプレートを格納するパス。

# -- Options for EPUB output
# EPUB出力に関するオプション。

epub_show_urls = "footnote"  # URLを脚注として表示。

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# ソースファイルを検索するときに無視するファイルやディレクトリのパターンのリスト。
# This pattern also affects html_static_path and html_extra_path.
# このパターンはhtml_static_pathやhtml_extra_pathにも影響します。
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]  # 無視するパターンを指定。

# -- Options for HTML output -------------------------------------------------
# HTML出力に関するオプション。

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# HTMLおよびHTMLヘルプページに使用するテーマを指定します。
# 組み込みテーマのリストはドキュメントを参照してください。
#
html_theme = "sphinx_rtd_theme"  # Read the Docsテーマを使用。

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# カスタム静的ファイル（スタイルシートなど）を含むパスをここに追加します。
# これらは組み込みの静的ファイルの後にコピーされるため、
# "default.css"という名前のファイルは組み込みの"default.css"を上書きします。
html_static_path = ["_static"]  # カスタム静的ファイルを格納するパス。

html_theme_options = {
    'analytics_anonymize_ip': False,  # IP匿名化を無効化。
    'logo_only': False,  # ロゴのみの表示を無効化。
    'display_version': True,  # バージョン番号を表示。
    'prev_next_buttons_location': 'bottom',  # 前後のボタンを下部に配置。
    'style_external_links': False,  # 外部リンクのスタイルを無効化。
    'vcs_pageview_mode': '',  # バージョン管理システムのページビュー設定を空にする。
    'style_nav_header_background': '#5F5CBF',  # ナビゲーションヘッダーの背景色を設定。
    # Toc options
    # 目次オプション。
    'collapse_navigation': True,  # ナビゲーションを折りたたむ。
    'sticky_navigation': True,  # ナビゲーションを固定する。
    'navigation_depth': 3,  # ナビゲーションの深さを設定。
    'includehidden': True,  # 隠しコンテンツを含める。
    'titles_only': False,  # タイトルのみの表示を無効化。
}
