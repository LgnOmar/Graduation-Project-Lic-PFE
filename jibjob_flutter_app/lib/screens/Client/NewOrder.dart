import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

import 'package:jibjob/screens/Offre.dart';

class NewOrder extends StatefulWidget {
  @override
  _NewOrderState createState() => _NewOrderState();
}

class _NewOrderState extends State<NewOrder> {
  final TextEditingController _descriptionController = TextEditingController();
  final TextEditingController _phoneController = TextEditingController();
  final TextEditingController _budgetController = TextEditingController();
  final List<String?> Liste_images = []; 

  final List<XFile?> _images = List<XFile?>.filled(6, null);
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(int index) async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _images[index] = image;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
  preferredSize: Size.fromHeight(70),
  child: AppBar(
    backgroundColor: Colors.white,
    elevation: 0,
    automaticallyImplyLeading: false,
    flexibleSpace: SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            ElevatedButton(
              onPressed: () {
                for (var item in _images) {
                  if (item != null) {
                    Liste_images.add(item.path);
                  }
                }

                Offre offre = Offre(
                  demande: _descriptionController.text,
                  phone: _phoneController.text,
                  address: _budgetController.text,
                  images: Liste_images ,
                );
                
                Liste_Offres.add(offre) ;

                print("length = ${Liste_Offres.length} \n\n") ;

                for (var item in Liste_Offres) {
                  print("Demande : ${item.demande}\n");
                  print("Demande : ${item.phone}\n");
                  print("Demande : ${item.address}\n");
                  print("Demande : ${item.images}\n\n\n");
                }

                Liste_images.clear() ;
 ;

                Navigator.pop(context);
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Color(0xFF7B61FF),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                elevation: 0,
              ),
              child: Text(
                'نشر الطلب',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                  fontFamily: 'Cairo', 
                ),
              ),
            ),
            Row(
              children: [
                Text(
                  'طلب زبون جديد',
                  style: TextStyle(
                    color: Color(0xFF20004E),
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                    fontFamily: 'Cairo', // Use your Arabic font if available
                  ),
                ),
                SizedBox(width: 8),
                GestureDetector(
                  onTap: () {
                    Navigator.of(context).pop();
                  },
                  child: Icon(
                    Icons.close,
                    color: Color(0xFF20004E),
                    size: 28,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    ),
  ),
),

      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [

              Container(
  margin: EdgeInsets.symmetric(vertical: 12),
  padding: EdgeInsets.symmetric(horizontal: 8, vertical: 10),
  decoration: BoxDecoration(
    color: Color(0xFFB8F9CC), // light green background
    borderRadius: BorderRadius.circular(6),
    border: Border.all(
      color: Color(0xFF2ECC71), // green border
      width: 2,
    ),
  ),
  child: Row(
    mainAxisAlignment: MainAxisAlignment.center,
    children: [
      Text(
        'خلي معلوماتك دقايق يتصلو بيك',
        style: TextStyle(
          color: Color(0xFF2ECC71), // green text
          fontWeight: FontWeight.w500,
          fontSize: 18,
          fontFamily: 'Cairo', // or your Arabic font
        ),
      ),
      SizedBox(width: 8),
      Icon(
        Icons.thumb_up_alt_outlined,
        color: Color(0xFF2ECC71),
      ),
    ],
  ),
) ,
              Text(
                'الطلب   ',
                style: TextStyle(fontWeight: FontWeight.bold ,fontSize: 17 , color: const Color(0xFF130160)),
              ),
              TextField(
                textAlign: TextAlign.right,
                controller: _descriptionController,
                maxLines: 3,
                decoration: InputDecoration(
                  hintText: 'اكتب وش تحتاج راح نبعثو الطلب تاعك للمختصين القراب و لي نوثقو فيهم دقايق يتصلو بيك',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 16),
              Text(
                'رقم الهاتف   ',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 17 , color: const Color(0xFF130160)),
              ),
              TextField(
                textAlign: TextAlign.right,
                controller: _phoneController,
                keyboardType: TextInputType.phone,
                decoration: InputDecoration(
                  hintText: 'باش يتصلو بيك',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 16),
              Text(
                'البلدية   ',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 17 , color: const Color(0xFF130160)),
              ),
              TextField(
                textAlign: TextAlign.right,
                controller: _budgetController,
                decoration: InputDecoration(
                  hintText: 'باش نبعثولك اقرب واحد',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 16),
              Text(
                'الصور   ',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 17 , color: const Color(0xFF130160) ),
              ),
              GridView.builder(
                shrinkWrap: true,
                physics: NeverScrollableScrollPhysics(),
                gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 3,
                  crossAxisSpacing: 8,
                  mainAxisSpacing: 8,
                ),
                itemCount: 6,
                itemBuilder: (context, index) {
                  return GestureDetector(
                    onTap: () => _pickImage(index),
                    child: Container(
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: _images[index] != null
                          ? Image.file(
                              File(_images[index]!.path),
                              fit: BoxFit.cover,
                            )
                          : Icon(Icons.camera_alt, color: Colors.grey),
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
