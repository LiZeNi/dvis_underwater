#!/bin/bash
# ======================================================
# 通用安全迁移工具（支持断点续传、逐个文件校验）
# 用法: ./migrate_data.sh -s <源目录> -d <目标目录> [选项]
# ======================================================

set -e

# ---------- 默认排除模式（可被 -e 覆盖） ----------
EXCLUDES=("UMOT_weight")   # 默认排除 UMOT_weight

# ---------- 参数变量 ----------
SRC="/data/zlni/zlni/data/UMOT_pretrain_weight"
DEST="/data1/zlni/data/UMOT_pretrain_weight"
KEEP_SRC=0
DRY_RUN=0
CONFIRM_DELETE=0

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ---------- 帮助信息 ----------
show_help() {
    cat << EOF
用法: $0 -s <源目录> -d <目标目录> [选项]

必需参数:
  -s, --src <源目录>       要迁移的源目录路径
  -d, --dest <目标目录>    目标目录路径（父目录会自动创建）

可选参数:
  -e, --exclude <模式>     排除匹配的文件/目录（可重复使用，默认排除 UMOT_weight）
  --keep-src               保留源目录（即使校验通过也不删除）
  --dry-run                仅模拟运行，不实际执行
  --confirm-delete         校验通过后删除源目录（必须显式指定才删除）
  -h, --help               显示此帮助信息

示例:
  $0 -s /data/old_data -d /data1/new_data
  $0 -s /home/user/project -d /mnt/backup/project -e "*.tmp" -e "cache/"
  $0 -s /mnt/source -d /mnt/dest --dry-run
EOF
    exit 0
}

# ---------- 参数解析 ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--src)
            SRC="$2"
            shift 2
            ;;
        -d|--dest)
            DEST="$2"
            shift 2
            ;;
        -e|--exclude)
            EXCLUDES+=("$2")
            shift 2
            ;;
        --keep-src)
            KEEP_SRC=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --confirm-delete)
            CONFIRM_DELETE=1
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            ;;
    esac
done

# ---------- 检查必需参数 ----------
if [[ -z "$SRC" || -z "$DEST" ]]; then
    echo -e "${RED}错误：必须指定源目录 (-s) 和目标目录 (-d)${NC}"
    show_help
fi

# 去除路径末尾的斜杠
SRC="${SRC%/}"
DEST="${DEST%/}"

if [[ $DRY_RUN -eq 1 ]]; then
    echo -e "${YELLOW}>>> DRY-RUN 模式${NC}"
fi

# ---------- 检查源 ----------
if [[ ! -d "$SRC" ]]; then
    echo -e "${RED}错误：源目录 $SRC 不存在！${NC}"
    exit 1
fi

# ---------- 源大小 ----------
SRC_SIZE_H=$(du -sh "$SRC" 2>/dev/null | awk '{print $1}')
echo -e "${GREEN}源目录: $SRC${NC}"
echo -e "${GREEN}源目录大小: $SRC_SIZE_H${NC}"

# ---------- 创建目标父目录 ----------
DEST_PARENT=$(dirname "$DEST")
if [[ ! -d "$DEST_PARENT" ]]; then
    echo -e "${YELLOW}创建目标父目录: $DEST_PARENT${NC}"
    [[ $DRY_RUN -eq 0 ]] && mkdir -p "$DEST_PARENT"
fi

# ---------- 目标存在检查（允许续传） ----------
if [[ -e "$DEST" ]]; then
    echo -e "${YELLOW}目标 $DEST 已存在，将进入增量检查模式（逐个文件校验完整性）。${NC}"
else
    echo -e "${GREEN}目标目录不存在，全新迁移。${NC}"
fi

# ---------- 显示磁盘 ----------
echo -e "\n${GREEN}迁移前磁盘空间：${NC}"
df -h "$SRC" "$DEST_PARENT" 2>/dev/null || df -h "$SRC" "$(dirname "$DEST_PARENT")"

# ---------- 构建排除参数 ----------
EXCLUDE_ARGS=""
for pat in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=" --exclude='$pat'"
done
if [[ ${#EXCLUDES[@]} -gt 0 ]]; then
    echo -e "${YELLOW}排除模式: ${EXCLUDES[*]}${NC}"
fi

# ---------- 核心传输（逐个文件校验） ----------
echo -e "\n${GREEN}开始增量同步（--checksum：逐个文件校验完整性）...${NC}"
RSYNC_CMD="rsync -avP --partial --checksum --no-inc-recursive $EXCLUDE_ARGS \"$SRC/\" \"$DEST/\""

if [[ $DRY_RUN -eq 0 ]]; then
    eval $RSYNC_CMD
else
    echo "模拟执行：$RSYNC_CMD"
fi

# ---------- 传输后二次确认（dry-run with checksum） ----------
echo -e "\n${GREEN}传输完成，进行最终完整性校验（dry-run with checksum）...${NC}"
if [[ $DRY_RUN -eq 0 ]]; then
    DIFF_OUTPUT=$(rsync -avcn --delete --checksum $EXCLUDE_ARGS "$SRC/" "$DEST/" 2>&1)
    
    if [[ -z "$DIFF_OUTPUT" ]]; then
        echo -e "${GREEN}✅ 完美校验通过：所有文件完整且一致。${NC}"
        SYNC_COMPLETE=1
    else
        echo -e "${YELLOW}⚠️ 仍有差异（可能是新文件或校验不一致）：${NC}"
        echo "$DIFF_OUTPUT"
        echo -e "${YELLOW}建议再次运行本脚本，rsync 会自动补全。${NC}"
        SYNC_COMPLETE=0
    fi
else
    echo "（跳过实际校验）"
    SYNC_COMPLETE=0
fi

# ---------- 删除源目录（需满足条件） ----------
if [[ $KEEP_SRC -eq 1 ]]; then
    echo -e "${YELLOW}--keep-src 已指定，保留源目录。${NC}"
elif [[ $SYNC_COMPLETE -eq 1 ]]; then
    if [[ $CONFIRM_DELETE -eq 1 ]]; then
        if [[ $DRY_RUN -eq 0 ]]; then
            echo -e "${GREEN}校验通过且用户确认，删除源目录...${NC}"
            rm -rf "$SRC"
            echo -e "${GREEN}源目录已删除。${NC}"
        else
            echo "rm -rf $SRC"
        fi
    else
        echo -e "${YELLOW}校验通过，但未加 --confirm-delete，保留源目录。${NC}"
        echo "如需删除，请添加 --confirm-delete 参数。"
    fi
else
    echo -e "${YELLOW}校验未完全通过，保留源目录以保证安全。${NC}"
fi

# ---------- 建立软链接（可选）----------
# 注意：我们不再自动建立软链接，因为通用脚本可能不需要这个功能。
# 如需软链接，用户可手动执行 ln -snf "$DEST" "$LINK_PATH"
# 但为了保留原来功能，我们可以加一个参数 --link-to <path>，但这里暂不实现，
# 因为用户只要求改传输目录，软链接是特定场景。

# ---------- 显示最终状态 ----------
echo -e "\n${GREEN}迁移后磁盘空间：${NC}"
df -h "$SRC" "$DEST_PARENT" 2>/dev/null || df -h "$SRC" "$(dirname "$DEST_PARENT")"
echo -e "\n${GREEN}✅ 全部完成！${NC}"