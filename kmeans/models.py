from django.db import models

# Create your models here.

class Wells(models.Model):
    w_id = models.AutoField(primary_key=True)
    w_uwi = models.CharField(max_length=19)
    w_name = models.CharField(max_length=100, blank=True, null=True)
    w_drillers_total_depth = models.FloatField(blank=True, null=True)
    w_operator = models.CharField(max_length=100, blank=True, null=True)
    w_current_status = models.CharField(max_length=100, blank=True, null=True)
    w_province = models.CharField(max_length=20, blank=True, null=True)
    w_class = models.CharField(max_length=100, blank=True, null=True)
    w_bottom_lng = models.FloatField(blank=True, null=True)
    w_bottom_lat = models.FloatField(blank=True, null=True)
    w_top_lng = models.FloatField(blank=True, null=True)
    w_top_lat = models.FloatField(blank=True, null=True)
    w_pad = models.CharField(max_length=5, blank=True, null=True)
    w_number_in_pad = models.CharField(max_length=3, blank=True, null=True)
    w_type = models.CharField(max_length=10, blank=True, null=True)
    w_injection_months = models.IntegerField(blank=True, null=True)
    w_producion_months = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'wells'