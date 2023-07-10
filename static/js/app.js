let flower = {"rose":"Rosaceae (Gülgiller) ailesinden olan gül, en çok hediye edilen, sevginin, aşkın ve saflığın simgesi olan bir çiçektir.",
      "daisy":"Çok yıllık bitki olmalarına rağmen tek yıllık işlemi uygulanan otsu bitkilerdir.",
      "dandelion":"Radika —doğrusu radikya— bu bitkiye Rumların verdiği isim ve genelde Ege bölgesinde bu adla biliniyor.",
      "sunflower":"Ayçiçeği; mutluluk, iyimserlik, dürüstlük, uzun ömür, sadakat gibi birçok anlama gelmektedir.",
      "tulip":"Lale çiçeği zarif, asil ve aşk anlamını ifade eder. Lale çiçeği insanlara rahatlık hissi uyandırır."};

document.querySelector('#desc').innerText= flower[document.querySelector('.data').innerText]

document.querySelector('.type').innerText = document.querySelector('.data').innerText