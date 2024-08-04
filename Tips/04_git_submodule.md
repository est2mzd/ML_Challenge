# GitHubでのサブフォルダ管理方法

GitHubで他のリポジトリをフォルダとして管理したい場合、以下の2つの方法が一般的です：

## 1. Git Submodule

Git Submoduleを使用すると、他のGitリポジトリを自分のリポジトリのサブディレクトリとして追加できます。これは、特定のバージョンを固定して使用するのに適しています。

- **追加手順**：
  ```bash
  git submodule add <repository_url> <path_to_directory>
  git submodule init
  git submodule update

- **更新手順**：サブモジュール内に移動し、通常のGit操作を行います。
  ```bash
    cd <path_to_directory>
    git pull origin <branch>


## 2. Git Subtree

Git Subtreeは、外部リポジトリのコンテンツをコピーして、自分のリポジトリに統合する方法です。こちらは、外部リポジトリの履歴を一緒に管理しやすく、サブモジュールのような管理の手間が少ないのが利点です。

- **追加手順**：
  ```bash
    git subtree add --prefix=<path_to_directory> <repository_url> <branch> --squash

- **更新手順**：
  ```bash
    git subtree pull --prefix=<path_to_directory> <repository_url> <branch> --squash

## 注意点

 - サブモジュール：外部リポジトリの変更を取得する際に明示的な操作が必要です。
 - サブツリー：外部リポジトリの履歴がマージされるため、リポジトリサイズが大きくなる可能性があります。
 
これらの方法の選択は、プロジェクトのニーズや管理のしやすさによって決まります。