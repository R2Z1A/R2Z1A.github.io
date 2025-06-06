// hexo.extend.filter.register("new_post_path", function (data) {
//     console.log(data);
//   });

// hexo.on("new", function(post){
//     console.log(post)
// });

// // 测试用
// // hexo.extend.filter.register("before_post_render", function (post){
// //     if (post.title.includes('test')){
// //         console.log(post)
// //     }
// // });

hexo.extend.filter.register("before_post_render", function (post) { 
    if (post.start_date) {
        const moment = require('moment');

        const startDate = moment(post.start_date);
        const endDate = moment(post.date).add(8, 'hours');

        const timeDifference = moment.duration(endDate.diff(startDate));
        
        // console.log(post.title)
        // console.log(startDate)
        // console.log(endDate)
        // console.log(timeDifference)

        // 将时间差转换为秒、分钟、小时、天
        const months = timeDifference.months();
        const days = timeDifference.days();
        const hours = timeDifference.hours();
        const minutes = timeDifference.minutes();
        const seconds = timeDifference.seconds();

        post.content = post.content.replace(
            '</@red_stone_time>',
            `${months}月${days}天${hours % 24}时${minutes % 60}分${seconds % 60}秒`
        );
        return post
    }
    return post
});