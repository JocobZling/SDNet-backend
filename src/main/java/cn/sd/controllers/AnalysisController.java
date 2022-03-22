package cn.sd.controllers;

import cn.sd.services.AnalysisService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;

@RestController
@RequestMapping(value = "/api/analysis")
public class AnalysisController {

    private final AnalysisService analysisService;

    @Autowired
    public AnalysisController(AnalysisService analysisService) {this.analysisService = analysisService;}

    @PostMapping("/hist")
    public ResponseEntity<?> analyse(@RequestParam(value = "file") MultipartFile file, HttpServletRequest request) throws Exception{
        return ResponseEntity.ok(analysisService.histImage(file));
    }
}
