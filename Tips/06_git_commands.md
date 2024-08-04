# よく使うGitコマンド

Gitは、バージョン管理を行うための強力なツールであり、多くのコマンドがあります。以下に、よく使われるGitコマンドを紹介します。

## よく使うGitコマンド

1. **`git init`**
   - リポジトリを初期化します。現在のディレクトリに新しいGitリポジトリを作成します。
   ```bash
      git init

2. **`git clone`**
   - リモートリポジトリをローカルにコピーします。
   ```bash
   git clone <repository_url>

3. **`git add`**
   - ファイルをステージングエリアに追加します。コミットの準備をします。
   ```bash
   git add <file_or_directory>

4. **`git commit`**
   - ステージングエリアの変更をリポジトリにコミットします。メッセージを追加することが重要です。
   ```bash
   git commit -m "commit message"
   
5. **`git status`**
   - 作業ツリーの状態を確認します。追跡されているファイルやステージングエリアの内容がわかります。
   ```bash
   git status

6. **`git log`**
   - コミット履歴を表示します。詳細な履歴を確認するのに役立ちます。
   ```bash
   git log

7. **`ｇit diff`**
   - 変更点を確認します。ワーキングディレクトリとステージングエリアの差分を表示します。
   ```bash
   git diff

8. **`git branch`**
   - ブランチの操作を行います。新しいブランチを作成したり、現在のブランチを確認できます。
   ```bash
   git branch  # 現在のブランチ一覧を表示
   git branch <branch_name>  # 新しいブランチを作成

9. **`git checkout`**
   - ブランチやコミットをチェックアウトします。ブランチの切り替えやファイルの復元に使われます。
   ```bash
   git checkout <branch_name>  # ブランチを切り替え
   git checkout <commit_hash>  # 特定のコミットをチェックアウト

10. **`git merge`**
   - ブランチをマージします。変更を統合します。
   ```bash
   git merge <branch_name>
   git checkout <commit_hash>  # 特定のコミットをチェックアウト

11. **`git pull`**
   - リモートリポジトリから変更を取得して、ローカルリポジトリを更新します。
   ```bash
   git pull origin <branch_name>

12. **`git push`**
   - ローカルリポジトリの変更をリモートリポジトリに送信します。
   ```bash
   git push origin <branch_name>

13. **`git reset`**
   - コミットをリセットします。HEADを特定の状態に戻します。
   ```bash
   git reset <commit_hash>

14. **`git stash`**
   - 作業中の変更を一時的に保存します。作業ツリーをクリーンにしたいときに使います。
   ```bash
   git stash
   git stash pop  # スタッシュした変更を再適用   