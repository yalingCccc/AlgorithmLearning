# 分步复现wechat大数据竞赛2021 top1方案
## 模型效果
## **版本1**
- 特征
   - user id
   - feed id
   - w2v同现embedding初始化feed id embedding 1
   - feed embedding初始化feed id embedding 2
- 网络
   - MMoE 
- 设置
   - epoch_num: 5
   - expert_num: 8
- AUC
   - 'score': 0.549608
   - 'score_detail': 
     - 'read_comment': 0.532772
     - 'like': 0.534525
     - 'click_avatar': 0.600776
     - 'forward': 0.559866
## **版本2**
- 特征
   - 利用全连接层融合feed emb 1和feed emb 2
   - 增加用户多行为历史列表
   - 增加用户单行为历史列表
   - 增加用户历史完成度列表
   - 增加展现但未交互历史队列
- 网络
   - DLRM：DIN+MMoE+cos交互网络
- 设置
   - epoch_num: 5
   - expert_num: 5
- AUC
   - 'score': 
   - 'score_detail': 
     - 'read_comment': 
     - 'like':
     - 'click_avatar': 
     - 'forward':

## **版本3**
- 特征
   - 增加context信息
   - 使用w2v同现emb初始化用户embedding
- AUC
   - 'score': 
   - 'score_detail': 
     - 'read_comment':
     - 'like':
     - 'click_avatar': 
     - 'forward': 
     - 
## **版本4**
- 特征
   - 增加side information
- AUC
   - 'score':
   - 'score_detail': 
     - 'read_comment': 
     - 'like': 
     - 'click_avatar': 
     - 'forward':

<table class="MsoTableGrid" border="1" cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes">
  <td width="66" valign="top" style="width:49.4pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span style="font-family:宋体;mso-ascii-font-family:
  &quot;Times New Roman&quot;;mso-hansi-font-family:&quot;Times New Roman&quot;">版本</span></p>
  </td>
  <td width="265" valign="top" style="width:7.0cm;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span style="font-family:宋体;mso-ascii-font-family:
  &quot;Times New Roman&quot;;mso-hansi-font-family:&quot;Times New Roman&quot;">描述</span></p>
  </td>
  <td width="142" valign="top" style="width:106.3pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US">AUC</span></p>
  </td>
  <td width="81" valign="top" style="width:60.65pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span style="font-family:宋体;mso-ascii-font-family:
  &quot;Times New Roman&quot;;mso-hansi-font-family:&quot;Times New Roman&quot;">提升</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1">
  <td width="66" valign="top" style="width:49.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p>1</span></p>
  </td>
  <td width="265" valign="top" style="width:7.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p>baseline</span></p>
  </td>
  <td width="142" valign="top" style="width:106.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p>0.55</span></p>
  </td>
  <td width="81" valign="top" style="width:60.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p>-</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2">
  <td width="66" valign="top" style="width:49.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p>2</span></p>
  </td>
  <td width="265" valign="top" style="width:7.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p>1、融合feed embedding
     2、增加历史队列3、优化网络</span></p>
  </td>
  <td width="142" valign="top" style="width:106.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
  <td width="81" valign="top" style="width:60.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3">
  <td width="66" valign="top" style="width:49.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
  <td width="265" valign="top" style="width:7.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
  <td width="142" valign="top" style="width:106.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
  <td width="81" valign="top" style="width:60.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;mso-yfti-lastrow:yes">
  <td width="66" valign="top" style="width:49.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
  <td width="265" valign="top" style="width:7.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" style="text-indent:0cm;mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
  <td width="142" valign="top" style="width:106.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
  <td width="81" valign="top" style="width:60.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center;text-indent:0cm;
  mso-char-indent-count:0"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
 </tr>
</tbody></table>