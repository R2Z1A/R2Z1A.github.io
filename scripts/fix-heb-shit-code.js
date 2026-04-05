// Hexo 官方注入功能：自动在页面头部添加修复代码
// 完全不碰主题/插件/node_modules，安全永久生效
hexo.extend.injector.register('head_begin', () => {
  return `
<script>
// 浏览器全局顶级对象，ES模块也能访问
globalThis.log = {
  info: function() {},
  warn: function() {},
  error: function() {}
};
globalThis.ensurePrefix = function(str) { return str; };
</script>
  `;
});