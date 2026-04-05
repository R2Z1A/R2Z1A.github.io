#!/usr/bin/env node
/**
 * GitHub Actions 解密脚本
 * 用于在 CI/CD 环境中解密加密的文章
 * 用法: node decrypt-workflow.js -i <输入路径> [-k <密钥>]
 * 密钥优先顺序: 1) 命令行参数 -k  2) 环境变量 DECRYPT_KEY  3) 环境变量 KEY
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// 解析命令行参数
function parseArgs() {
    const args = process.argv.slice(2);
    const options = {
        input: null,
        key: null,
        directory: false,
        removeSource: false,
        ext: '.encrypted'
    };

    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '-i':
            case '--input':
                options.input = args[++i];
                break;
            case '-k':
            case '--key':
                options.key = args[++i];
                break;
            case '-d':
            case '--directory':
                options.directory = true;
                break;
            case '-r':
            case '--remove':
                options.removeSource = true;
                break;
            case '--ext':
                options.ext = args[++i];
                break;
            case '-h':
            case '--help':
                showHelp();
                process.exit(0);
        }
    }

    if (!options.input) {
        console.error('❌ 错误: 必须提供输入文件或目录 (-i)');
        showHelp();
        process.exit(1);
    }

    return options;
}

function showHelp() {
    console.log(`
🔓 GitHub Actions 解密工具

用法: node decrypt-workflow.js -i <输入路径> [选项]

选项:
  -i, --input <路径>    输入文件或目录（必需）
  -k, --key <密钥>      解密密钥（可选，默认从环境变量读取）
  -d, --directory      处理整个目录
  -r, --remove         解密后删除原始加密文件
  --ext <扩展名>        加密文件的扩展名（默认 .encrypted）
  -h, --help           显示帮助信息

密钥来源优先级:
  1. 命令行参数 -k
  2. 环境变量 DECRYPT_KEY
  3. 环境变量 KEY

示例:
  # 解密单个文件（从环境变量读取密钥）
  node decrypt-workflow.js -i source/_posts/my-post.md.encrypted

  # 解密整个目录并删除加密文件
  node decrypt-workflow.js -i source/_posts/ -d -r
    `);
}

// 获取密钥（优先级: 参数 > DECRYPT_KEY > KEY）
function getKey(keyFromArgs) {
    if (keyFromArgs) {
        console.log('✓ 使用命令行提供的密钥');
        return keyFromArgs;
    }
    if (process.env.DECRYPT_KEY) {
        console.log('✓ 使用环境变量 DECRYPT_KEY');
        return process.env.DECRYPT_KEY;
    }
    if (process.env.KEY) {
        console.log('✓ 使用环境变量 KEY');
        return process.env.KEY;
    }
    return null;
}

// Fernet-like 解密 (AES-128-CBC with HMAC)
function decryptData(encryptedData, key) {
    const keyBuffer = Buffer.from(key, 'base64');
    const data = Buffer.from(encryptedData, 'base64');

    // 解析格式: version(1) + timestamp(8) + iv(16) + hmac(32) + ciphertext
    const version = data[0];
    if (version !== 0x80) {
        throw new Error('不支持的加密版本');
    }

    const iv = data.slice(9, 25);
    const hmacValue = data.slice(25, 57);
    const ciphertext = data.slice(57);

    // 验证 HMAC
    const hmac = crypto.createHmac('sha256', keyBuffer);
    hmac.update(iv);
    hmac.update(ciphertext);
    const computedHmac = hmac.digest();

    if (!crypto.timingSafeEqual(hmacValue, computedHmac)) {
        throw new Error('HMAC 验证失败：密钥错误或数据被篡改');
    }

    // 解密
    const decipher = crypto.createDecipheriv('aes-128-cbc', keyBuffer.slice(0, 16), iv);
    let decrypted = decipher.update(ciphertext);
    decrypted = Buffer.concat([decrypted, decipher.final()]);

    return decrypted;
}

// 解密单个文件
function decryptFile(inputFile, outputFile, key, removeSource = false) {
    const encryptedData = fs.readFileSync(inputFile, 'utf8');
    const decrypted = decryptData(encryptedData, key);
    fs.writeFileSync(outputFile, decrypted);
    console.log(`✅ 已解密: ${inputFile} -> ${outputFile}`);

    if (removeSource) {
        fs.unlinkSync(inputFile);
        console.log(`🗑️  已删除源文件: ${inputFile}`);
    }
}

// 处理目录
function processDirectory(directory, key, ext, removeSource = false) {
    const decryptedFiles = [];

    function walkDir(dir) {
        const files = fs.readdirSync(dir);
        for (const file of files) {
            const filePath = path.join(dir, file);
            const stat = fs.statSync(filePath);

            if (stat.isDirectory()) {
                walkDir(filePath);
            } else if (file.endsWith(ext)) {
                const outputPath = filePath.slice(0, -ext.length);
                decryptFile(filePath, outputPath, key, removeSource);
                decryptedFiles.push(outputPath);
            }
        }
    }

    walkDir(directory);
    return decryptedFiles;
}

// 主函数
function main() {
    console.log('🔓 GitHub Actions 解密工具启动...\n');

    const options = parseArgs();

    // 加载密钥（支持文件路径或直接传入）
    if (options.key) {
        options.key = loadKey(options.key);
    }

    // 获取密钥（优先级: 参数 > DECRYPT_KEY > KEY）
    const key = getKey(options.key);
    if (!key) {
        console.error('❌ 错误: 未提供密钥，请使用 -k 参数或设置环境变量 DECRYPT_KEY/KEY');
        process.exit(1);
    }

    // 执行解密
    console.log('\n开始解密...');

    try {
        if (options.directory) {
            const decrypted = processDirectory(options.input, key, options.ext, options.removeSource);
            console.log(`\n✅ 共解密 ${decrypted.length} 个文件`);
        } else {
            const output = options.output || options.input.replace(options.ext, '');
            decryptFile(options.input, output, key, options.removeSource);
        }

        console.log('\n✅ 解密完成！');
    } catch (error) {
        console.error(`\n❌ 解密失败: ${error.message}`);
        process.exit(1);
    }
}

// 从文件或字符串获取密钥
function loadKey(keyOrFilePath) {
    // 如果看起来像文件路径且文件存在，则读取文件
    if (keyOrFilePath.includes('/') || keyOrFilePath.includes('\\')) {
        try {
            if (fs.existsSync(keyOrFilePath)) {
                const keyFromFile = fs.readFileSync(keyOrFilePath, 'utf8').trim();
                console.log(`✓ 从文件加载密钥: ${keyOrFilePath}`);
                return keyFromFile;
            }
        } catch (err) {
            // 读取文件失败，直接使用传入的值作为密钥
        }
    }
    // 直接使用传入的值作为密钥
    return keyOrFilePath;
}

// 运行
main();
