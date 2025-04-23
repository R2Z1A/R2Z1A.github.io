hexo.extend.filter.register("new_post_path", function (data) {
    console.log(data);
  });

hexo.on("new", function(post){
    console.log(post)
});

// 测试用
// hexo.extend.filter.register("before_post_render", function (post){
//     if (post.title.includes('test')){
//         console.log(post)
//     }
// });

hexo.extend.filter.register("before_post_render", function (post) {    
    if (post.start_date) {
        const moment = require('moment');

        const startDate = moment(post.start_date);
        const endDate = post.date;

        const timeDifference = moment.duration(endDate.diff(startDate));

        // 将时间差转换为秒、分钟、小时、天
        const days = timeDifference.days();
        const hours = timeDifference.hours();
        const minutes = timeDifference.minutes();
        const seconds = timeDifference.seconds();

        post.content = post.content.replace(
            '</@red_stone_time>',
            `${days}天${hours % 24}时${minutes % 60}分${seconds % 60}秒`
        );
        return post
    }
});