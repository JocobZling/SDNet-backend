CREATE TABLE `newdetection` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `pictureOnePosition` VARCHAR(200),
    `pictureTwoPosition` VARCHAR(200),
    `originalImagePosition` VARCHAR(200),
    `fileName` VARCHAR(200),
    `result` VARCHAR(200),
    `userId` int(11),
    `typeId` int(11),
    `startTime` TIMESTAMP default current_timestamp,
    `endTime` TIMESTAMP default current_timestamp,
    `createTime` TIMESTAMP default current_timestamp,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
