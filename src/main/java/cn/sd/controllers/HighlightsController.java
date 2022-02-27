package cn.sd.controllers;

import cn.sd.entities.Photo;
import cn.sd.entities.vo.PhotoDisplayVo;
import cn.sd.entities.vo.TimeAndPosition;
import cn.sd.services.HighLightsService;
import cn.sd.services.PhotoService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

@RestController
@RequestMapping(value = "/api/highlight")
public class HighlightsController {

    private final PhotoService photoService;
    private final HighLightsService highLightsService;

    @Autowired
    public HighlightsController(PhotoService photoService, HighLightsService highLightsService) {
        this.photoService = photoService;
        this.highLightsService = highLightsService;
    }

    //封面一张 所有
    @GetMapping(value = "/{userId}")
    public ResponseEntity<?> getHighlightPageContent(@PathVariable Long userId) {
        // 先拿到每天里最精彩的第一张
        List<HashMap<String, String>> oneTimeOnePhoto = highLightsService.getEverytimePhoto(userId);
        if (oneTimeOnePhoto.size() == 0) {
            return new ResponseEntity<>("您还没有上传图片！", HttpStatus.OK);
        }
        // 在拿到person 分数最好的一张
        String bestP = highLightsService.BsetPerson(userId);
        //在拿到所有分数最好的一张
        String bestA = highLightsService.BsetAll(userId);
        // 最近face分数最好的
        String bestRecentP = highLightsService.BestRecentPerson(userId);
        // 最近all分数最好的
        String bestRecentA = highLightsService.BestRecentAll(userId);

        HashMap<String, TimeAndPosition> result = new HashMap<String, TimeAndPosition>();
        result.put("EveryTime", new TimeAndPosition(oneTimeOnePhoto, null));
        result.put("BestPerson", new TimeAndPosition(null, bestP));
        result.put("BestAll", new TimeAndPosition(null, bestA));
        result.put("BestRecentPerson", new TimeAndPosition(null, bestRecentP));
        result.put("BestRecentAll", new TimeAndPosition(null, bestRecentA));
        return new ResponseEntity<>(result, HttpStatus.OK);
    }


    @GetMapping(value = "/detail/{type}/{userId}")
    public ResponseEntity<?> getHighlightDetail(@PathVariable String type, @PathVariable String userId, @RequestParam(required = false) String date) {
        Long id = Long.parseLong(userId);
        switch (type) {
            case "EveryTime":
                return getEveryTimesAllHighlight(id, date);
            case "BestPerson":
                return getTopPersonHighlight(id);
            case "BestAll":
                return getTopAllHighlight(id);
            case "BestRecentPerson":
                return getRecentPersonHighlight(id);
            case "BestRecentAll":
                return getRecentAllHighlight(id);
            default:
                return new ResponseEntity<>(null, HttpStatus.BAD_REQUEST);
        }
    }

    //    //拿出用户这一天的所有分数的top10
//    @GetMapping(value = "/everyTimes/{createTime}/all/{userId}")
    public ResponseEntity<?> getEveryTimesAllHighlight(Long userId, String createTime) {
        int num = 10;
        List<PhotoDisplayVo> result = highLightsService.getTimeTop(userId, createTime, num);
        if (result.size() == 0) {
            return new ResponseEntity<>("没有当日图片！", HttpStatus.OK);
        }
        return new ResponseEntity<>(result, HttpStatus.OK);
    }

    //    //拿出用户所有的top person_score 10
//    @GetMapping(value = "/top/person/{userId}")
    public ResponseEntity<?> getTopPersonHighlight(Long userId) {
        // 根据userid取出这个所有的position以及对应的faceScore 根据faceScore排序
        // 然后给出position

        //需要取出的数量
        int num = 10;
        //flag = true 代表只按照faceScore排序
        boolean flag = true;
        List<PhotoDisplayVo> result = highLightsService.getTopPerson(userId, num, flag);
        if (result.size() == 0) {
            return new ResponseEntity<>("您还没有上传图片！", HttpStatus.OK);
        }
        return new ResponseEntity<>(result, HttpStatus.OK);
    }

    //    //拿出用户所有的top 所有的 10
//    @GetMapping(value = "/top/all/{userId}")
    public ResponseEntity<?> getTopAllHighlight(Long userId) {
        // 根据userid取出这个所有的position以及对应的总分数 根据总分数排序
        // 然后给出position

        //需要取出的数量
        int num = 10;
        //flag = false 代表按照总分数排序
        boolean flag = false;
        List<PhotoDisplayVo> result = highLightsService.getTopPerson(userId, num, flag);
        if (result.size() == 0) {
            return new ResponseEntity<>("您还没有上传图片！", HttpStatus.OK);
        }
        return new ResponseEntity<>(result, HttpStatus.OK);

    }

    //    //拿出用户最近的五天top person_scores 20
//    @GetMapping(value = "/recent/person/{userId}")
    public ResponseEntity<?> getRecentPersonHighlight(Long userId) {
        int num = 5; //最近几天
        int photoNum = 20; //图片数量
        //flag = true 代表只按照faceScore排序
        boolean flag = true;
        List<PhotoDisplayVo> result = highLightsService.getRecentTop(userId, num, flag, photoNum);
        if (result.size() == 0) {
            return new ResponseEntity<>("您还没有上传图片！", HttpStatus.OK);
        }

        return new ResponseEntity<>(result, HttpStatus.OK);
    }

    //    //拿出用户最近的五天top all_scores top 20
//    @GetMapping(value = "/recent/all/{userId}")
    public ResponseEntity<?> getRecentAllHighlight(Long userId) {
        int num = 5; //最近几天
        int photoNum = 20; //图片数量
        //flag = flase 代表只按照所有的分数排序
        boolean flag = false;
        List<PhotoDisplayVo> result = highLightsService.getRecentTop(userId, num, flag, photoNum);
        if (result.size() == 0) {
            return new ResponseEntity<>("您还没有上传图片！", HttpStatus.OK);
        }

        return new ResponseEntity<>(result, HttpStatus.OK);

    }

    @GetMapping(value = "test/{userId}")
    public ResponseEntity<?> test(@PathVariable Long userId) throws ParseException {
        List<Photo> result = highLightsService.test(userId);
        Date createTime1 = result.get(0).getCreateTime();
        Date createTime2 = result.get(2).getCreateTime();
        DateFormat df = new SimpleDateFormat("yyyy-MM-dd");
        String startDateStr = df.format(createTime1);
        Date startDate = df.parse(startDateStr);
        System.out.println(startDate);
        System.out.println(startDateStr);
        System.out.println(createTime2);
        return new ResponseEntity<>(result, HttpStatus.OK);
    }

}
