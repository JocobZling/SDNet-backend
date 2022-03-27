package cn.sd.controllers;

import cn.sd.entities.Detection;
import cn.sd.entities.User;
import cn.sd.exceptions.BusinessException;

import cn.sd.services.HistoryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.UnsupportedEncodingException;
import java.sql.SQLException;
import java.util.HashMap;
//import java.awt.print.Pageable;

@RestController
@RequestMapping(value = "/api/history")

public class HistoryController {


    private final HistoryService historyService;

    // 构造方法
    @Autowired
    public HistoryController(HistoryService historyService) {
        this.historyService = historyService;
    }


//    public ResponseEntity getFaceHistoryByUserId(@RequestBody FaceHistory History) {
//        History hhistory = historyService.findhistory(History);
//        return new ResponseEntity<>(hhistory, HttpStatus.OK);
//    }

//    public ResponseEntity<HashMap<String, Object>> getFaceHistoryByUserId(@RequestBody Detection History) throws BusinessException, UnsupportedEncodingException {
//        return ResponseEntity.ok(HistoryService.findhistory(History.getUserId()));

    @GetMapping(value = "/face/pageable/{userId}")
    public ResponseEntity getFaceHistoryByUserId(@PageableDefault(sort = {"id"}, direction = Sort.Direction.DESC) Pageable pageable,
                                                 @PathVariable(required = false) Long userId) {
        Page basicPage = historyService.getObjectivePageable(pageable, userId);
        return new ResponseEntity<>(basicPage, HttpStatus.OK);
    }
}



