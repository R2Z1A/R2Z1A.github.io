#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🔍 检查是否有明文 .md 文件...\n');

// 检查是否有明文的 .md 文件（不包含 .encrypted 或不是加密文件）
function checkPlaintextMdFiles() {
    const sourceDir = path.join(__dirname, '..', 'source');
    const hasPlaintextFiles = [];

    // 获取即将提交的所有文件
    const committedFiles = execSync('git diff --cached --name-only', { encoding: 'utf8' }).split('\n').filter(file => file.trim() !== '');

    // 检查即将提交的文件中是否有 source 目录下的明文 .md 文件
    for (const file of committedFiles) {
        const filePath = path.join(__dirname, '..', file);

        // 检查是否在 source 目录下且是 .md 文件且不是 .encrypted 文件
        if (file.startsWith('source/') && file.endsWith('.md') && !file.endsWith('.encrypted')) {
            hasPlaintextFiles.push(file);
        }
    }

    return hasPlaintextFiles;
}

const plaintextFiles = checkPlaintextMdFiles();

if (plaintextFiles.length > 0) {
    console.error('❌ 发现明文 .md 文件，禁止提交！\n');
    console.error('以下文件需要加密后才能提交：\n');
    plaintextFiles.forEach(file => {
        console.error(`  - ${file}`);
    });
    console.error('\n请运行加密命令后再提交：');
    console.error('  node encrypt.js -i source/_posts/ -d');
    process.exit(1);
} else {
    console.log('✅ 未发现明文 .md 文件，可以安全提交！');
    process.exit(0);
}
