const fs = require('fs');
const path = require('path');

const legacyDir = path.join(__dirname, '../source/_posts/legacy_private');
const files = fs.readdirSync(legacyDir).filter(f => f.endsWith('.md'));

console.log(`Found ${files.length} files in legacy_private`);

files.forEach(file => {
    const filePath = path.join(legacyDir, file);
    let content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n');

    let inFrontMatter = false;
    let i = 0;
    let backup = '';
    let newLines = [];

    while (i < lines.length) {
        const line = lines[i];

        if (line.trim() === '---') {
            inFrontMatter = !inFrontMatter;
            newLines.push(line);
            i++;
            continue;
        }

        if (inFrontMatter) {
            if (line.startsWith('tags:')) {
                backup += line + '\n';
                newLines.push('tags:');
                i++;
                while (i < lines.length && lines[i].startsWith('  - ')) {
                    backup += lines[i] + '\n';
                    i++;
                }
                continue;
            }

            if (line.startsWith('categories:')) {
                backup += line + '\n';
                newLines.push('categories:');
                i++;
                while (i < lines.length && lines[i].startsWith('  - ')) {
                    backup += lines[i] + '\n';
                    i++;
                }
                continue;
            }
        }

        newLines.push(line);
        i++;
    }

    if (backup) {
        newLines.push('');
        newLines.push('<!--');
        newLines.push('Original:');
        newLines.push(backup.trim());
        newLines.push('-->');

        const newContent = newLines.join('\n');
        fs.writeFileSync(filePath, newContent, 'utf8');
        console.log(`  ${file} -> Updated`);
    }
});

console.log(`\nDone!`);



