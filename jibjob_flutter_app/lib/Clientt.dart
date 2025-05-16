import 'package:flutter/material.dart';
import 'screens/JibJobAuthPage.dart';

// ignore: must_be_immutable
class Clientt extends StatelessWidget{

  TextEditingController emailC , mdpC , nameC , telC , communeC ;

  Clientt(this.emailC , this.mdpC , this.nameC , this.telC , this.communeC) ;

 @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor:  Colors.amber[50],

      body : Column( children: [
        SizedBox(height: 80) ,
        Container(alignment: Alignment.topCenter , child : Text( style : TextStyle(fontSize: 40 , fontWeight: FontWeight.bold) , "User Info\n")) ,
        Container(alignment: Alignment.topLeft , child : Text( style : TextStyle(fontSize: 15 , fontWeight: FontWeight.bold) , "  Email : ${emailC.text}\n")) ,
        Container(alignment: Alignment.topLeft , child : Text( style : TextStyle(fontSize: 15 , fontWeight: FontWeight.bold) , "  Password : ${mdpC.text}\n")) ,
        Container(alignment: Alignment.topLeft , child : Text( style : TextStyle(fontSize: 15 , fontWeight: FontWeight.bold) , "  Full Name : ${nameC.text}\n")) ,
        Container(alignment: Alignment.topLeft , child : Text( style : TextStyle(fontSize: 15 , fontWeight: FontWeight.bold) , "  Phone : ${telC.text}\n")) ,
        Container(alignment: Alignment.topLeft , child : Text( style : TextStyle(fontSize: 15 , fontWeight: FontWeight.bold) , "  City : ${communeC.text}\n\n\n")) ,

        Align(
              alignment: Alignment.center,
              child: Container(
                margin : EdgeInsets.fromLTRB(0, 0, 40, 0) ,
                height: 70,
                width: 70,
                decoration: BoxDecoration(
                  color: Colors.deepPurple,
                  shape: BoxShape.circle,
                ),
                child: IconButton(
                  icon: Icon(Icons.arrow_forward, color: Colors.white),
                  onPressed: () {
                      runApp(
                      MaterialApp(
                        home : JibJobAuthPage() ,
                        debugShowCheckedModeBanner: false
                      )) ;
                  },
                ),
              ),
            )





      ],
      )

    );
    
  }
}