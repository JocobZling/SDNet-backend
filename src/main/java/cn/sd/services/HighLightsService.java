package cn.sd.services;

import cn.sd.entities.Photo;
import cn.sd.entities.vo.PhotoDisplayVo;
import cn.sd.repositories.PhotoRepository;
import cn.sd.utils.SortUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

@Service
public class HighLightsService {
    private final PhotoRepository photoRepository;
    private final FaceService faceService;

    @Autowired
    public HighLightsService(PhotoRepository photoRepository, FaceService faceService) {
        this.photoRepository = photoRepository;
        this.faceService = faceService;
    }


    public List<PhotoDisplayVo> getTopPerson(Long userId, int num, boolean flag) {
        List<Photo> result = new ArrayList<>();
        List<Photo> resultPhoto = photoRepository.findPhotoByUserIdOrderByCreateTimeDesc(userId);
        if (resultPhoto.size() == 0) {
            return faceService.getPhotoDisplayList(result);
        }
        //调用排序
        List<Photo> SortList = SortUtil.SortPhoto(resultPhoto, flag);

        int len = SortList.size();
        if (len > num) {
            for (int i = 0; i < num; i++) {
                Photo tmp = SortList.get(i);
                result.add(tmp);
            }

        } else {
            result.addAll(SortList);
        }
        return faceService.getPhotoDisplayList(result);
    }

    // num是指定拿出最近几天的数据 flag是指定facescore排序还是allscore排序 photoNum是指定最后取出多少张
    public List<PhotoDisplayVo> getRecentTop(Long userId, int num, boolean flag, int photoNum) {
        List<Photo> result = new ArrayList<>();
        List<Photo> resultPhoto = photoRepository.findPhotoByUserIdOrderByCreateTimeDesc(userId);
        //拿到最近num 天的数据 不够num全拿来
        List<Photo> RencntList = SortUtil.GetNumPhotoBaseTime(resultPhoto, num);
        if (RencntList.size() == 0) {
            return faceService.getPhotoDisplayList(result);
        }
        //根据分数排序
        List<Photo> SortRecentList = SortUtil.SortPhoto(RencntList, flag);

        int len = SortRecentList.size();
        if (len > photoNum) {
            for (int i = 0; i < photoNum; i++) {
                Photo tmp = SortRecentList.get(i);
                result.add(tmp);
            }

        } else {
            result.addAll(SortRecentList);
        }
        return faceService.getPhotoDisplayList(result);


    }

    public List<PhotoDisplayVo> getTimeTop(long userId, String time, int num) {
        List<Photo> result = new ArrayList<>();
        List<Photo> resultPhoto = photoRepository.findPhotoByUserIdOrderByCreateTimeDesc(userId);
        List<Photo> timePhoto = SortUtil.GetTimePhoto(resultPhoto, time);
        if (timePhoto.size() == 0) {
            return faceService.getPhotoDisplayList(result);
        }
        boolean flag = false;

        List<Photo> sortTimePhoto = SortUtil.SortPhoto(timePhoto, flag);
        int len = sortTimePhoto.size();
        if (len > num) {
            for (int i = 0; i < num; i++) {
                Photo tmp = sortTimePhoto.get(i);
                result.add(tmp);
            }

        } else {
            result.addAll(sortTimePhoto);
        }

        return faceService.getPhotoDisplayList(result);
    }

    public List<HashMap<String, String>> getEverytimePhoto(long userId) {

        List<HashMap<String, String>> res = new ArrayList<HashMap<String, String>>();

        List<Photo> resultPhoto = photoRepository.findPhotoByUserIdOrderByCreateTimeDesc(userId);
        if (resultPhoto.size() == 0) {
            return res;
        }
        List<Photo> sortTimePhoto = SortUtil.SortTime(resultPhoto);

        List<List<Photo>> everyTimePhoto = SortUtil.GetEveryTimePhoto(sortTimePhoto);

        for (List<Photo> oneTimePhoto : everyTimePhoto) {
            HashMap<String, String> TimePosition = new HashMap<String, String>();
            List<Photo> tmp = SortUtil.SortPhoto(oneTimePhoto, false);
            TimePosition.put("date", SortUtil.DateToStringDay(tmp.get(0).getCreateTime()));
            TimePosition.put("src", tmp.get(0).getPosition());
            res.add(TimePosition);
        }
        return res;

    }

    public String BsetPerson(long userId) {
        List<PhotoDisplayVo> res = getTopPerson(userId, 1, true);
        return res.get(0).getSrc();
    }

    public String BsetAll(long userId) {
        List<PhotoDisplayVo> res = getTopPerson(userId, 1, false);
        return res.get(0).getSrc();
    }

    public String BestRecentPerson(long userId) {
        List<PhotoDisplayVo> res = getRecentTop(userId, 5, true, 1);
        return res.get(0).getSrc();
    }

    public String BestRecentAll(long userId) {
        List<PhotoDisplayVo> res = getRecentTop(userId, 5, false, 1);
        return res.get(0).getSrc();
    }


    public List<Photo> test(Long userId) {
        return photoRepository.findPhotoByUserIdOrderByCreateTimeDesc(userId);
    }

}
